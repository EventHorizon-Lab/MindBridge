"""Checks that one derived vector earns one fused contribution, from derivation to rank."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import cast

from mindbridge.application.capabilities import (
    Embedder,
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    ModelInput,
    TextPart,
)
from mindbridge.application.derive_observation_graph import (
    derive_observation_graph,
    embed_observation_graph,
)
from mindbridge.application.perception import (
    EventPerception,
    PerceivedEntity,
    PerceivedEvent,
)
from mindbridge.application.ports import (
    Answerer,
    EmbeddingIndex,
    EmbeddingMatch,
    EmbeddingSearch,
    MediaUrlSigner,
    MemoryStore,
    OccurrenceVerifier,
)
from mindbridge.application.recall import RecallMemories
from mindbridge.contracts import RecallQuery, RecallRequest
from mindbridge.core import (
    DeviceId,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    EntityType,
    Event,
    EventId,
    EvidenceId,
    MediaObjectId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelReference,
    Observation,
    ObservationId,
    Relation,
    RelationType,
    SensorKind,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
TENANT = TenantId("tenant_fusion")
MODEL = ModelReference(model_id="qwen3.8-max")
EMBEDDING_MODEL = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
SPACE = EmbeddingSpaceReference(space_id="jina-v5")
QUERY_VECTOR = (1.0, 0.0, 0.0)
# The observed event's vector, less like the query than the remembered memory's below it.
DERIVED_VECTOR = (0.8, 0.6, 0.0)
REMEMBERED_VECTOR = (0.9, math.sqrt(1.0 - 0.81), 0.0)
# The media vector of the clip the observed span was cut from: a different object, encoded
# from different bytes, and the one channel that is genuinely independent of the summary.
EVIDENCE_VECTOR = (0.95, math.sqrt(1.0 - 0.9025), 0.0)
# A second observed event, closer to the query than the first one, and the name of the
# entity the first one mentions: the query asked about that name, so the entity's own
# vector is the closest thing in the graph index to it.
NEIGHBOUR_VECTOR = (0.85, math.sqrt(1.0 - 0.7225), 0.0)
ENTITY_VECTOR = (0.99, math.sqrt(1.0 - 0.9801), 0.0)


def _input_text(model_input: ModelInput) -> str:
    return "".join(part.text for part in model_input.parts if isinstance(part, TextPart))


_PLACED_TOOL = PerceivedEvent(
    start_ms=500,
    end_ms=3_500,
    description="A person places a red tool beside a blue toolbox.",
    salience=0.8,
    evidence_ids=(EvidenceId("evidence_01"),),
)
# The same clip, with the person named: the entity carries a vector of its own, and it is
# the only object in the graph index that the query's name is close to.
_PLACED_TOOL_BY_MARA = replace(
    _PLACED_TOOL,
    entities=(
        PerceivedEntity(
            entity_type=EntityType.PERSON,
            canonical_name="Mara",
            confidence=0.9,
            evidence_ids=(EvidenceId("evidence_01"),),
        ),
    ),
)
_OPENED_TOOLBOX = PerceivedEvent(
    start_ms=4_000,
    end_ms=7_500,
    description="A person opens the blue toolbox and looks inside it.",
    salience=0.7,
    evidence_ids=(EvidenceId("evidence_03"),),
)


class FixedEmbedder:
    """Return one configured unit vector per input, or a different one for named text."""

    model_reference = EMBEDDING_MODEL
    space_reference = SPACE

    def __init__(
        self,
        values: tuple[float, ...],
        by_text: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        self.values = values
        self.by_text = by_text or {}

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        return EmbedResult(
            tuple(
                Embedding(
                    self.by_text.get(_input_text(model_input), self.values),
                    self.model_reference,
                    self.space_reference,
                )
                for model_input in request.inputs
            )
        )


class FakeEmbeddingIndex:
    """Rank stored vectors by cosine, exactly as the pgvector index does."""

    def __init__(self, embeddings: tuple[EmbeddingRecord, ...]) -> None:
        self.embeddings = embeddings

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        matches = [
            EmbeddingMatch(
                embedding_id=embedding.embedding_id,
                object_type=embedding.object_type,
                object_id=embedding.object_id,
                similarity=sum(
                    left * right
                    for left, right in zip(search.values, embedding.values, strict=True)
                ),
            )
            for embedding in self.embeddings
            if embedding.object_type in search.object_types
        ]
        ranked = sorted(
            (match for match in matches if match.similarity >= search.minimum_similarity),
            key=lambda match: -match.similarity,
        )
        return tuple(ranked[: search.limit])


class FakeMemoryStore:
    """Resolve each retrieval channel the way the PostgreSQL queries behind it do."""

    def __init__(
        self,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[Relation, ...],
    ) -> None:
        self.memories = {memory.memory_id: memory for memory in memories}
        self.relations = relations
        self.requested_memory_ids: list[tuple[MemoryId, ...]] = []

    async def search_memories_by_evidence(
        self,
        request: RecallRequest,
        ranked_evidence_ids: tuple[EvidenceId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return self._ranked(
            tuple(
                memory.memory_id
                for evidence_id in ranked_evidence_ids
                for memory in self.memories.values()
                if evidence_id in memory.evidence_ids
            ),
            limit=limit,
        )

    async def search_memories_by_ids(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        self.requested_memory_ids.append(ranked_memory_ids)
        return self._ranked(ranked_memory_ids, limit=limit)

    async def search_memories_by_hierarchy(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        # The recursive query seeds itself with the matched memories at depth 0, so every
        # matched memory comes back from this channel as well as from the ID lookup.
        return self._ranked(ranked_memory_ids, limit=limit)

    async def search_memories_by_graph_objects(
        self,
        request: RecallRequest,
        ranked_objects: tuple[EmbeddingMatch, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        # An entity match enters the walk through the events and claims that mention it, and
        # the query groups every path that reaches one memory into a single row carrying its
        # best rank, so a memory two matched objects both reach is returned once.
        represented = {
            (relation.source_type.value, relation.source_id): MemoryId(relation.target_id)
            for relation in self.relations
            if relation.relation_type is RelationType.REPRESENTED_BY
        }
        mentioning: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        for relation in self.relations:
            if relation.relation_type in {RelationType.MENTIONS, RelationType.ABOUT}:
                mentioning[relation.target_id].append(
                    (relation.source_type.value, relation.source_id)
                )
        found = tuple(
            memory_id
            for match in ranked_objects
            for source in (
                mentioning[match.object_id]
                if match.object_type is EmbeddedObjectType.ENTITY
                else [(match.object_type.value, match.object_id)]
            )
            if (memory_id := represented.get(source)) is not None
        )
        return self._ranked(tuple(dict.fromkeys(found)), limit=limit)

    def _ranked(
        self,
        memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories
        )[:limit]


async def test_a_derived_memory_is_credited_once_for_the_vector_it_shares() -> None:
    """Equal evidence must rank by similarity, not by how many keys one vector is filed under."""
    observed, embeddings, relations = await _observed_memory()
    remembered = _remembered_memory()
    store = FakeMemoryStore(
        (observed, remembered),
        relations,
    )
    recall = _recall(store, (*embeddings, _remembered_embedding(remembered)))

    fused = await recall._search_semantic_memories(
        RecallRequest(tenant_id=TENANT, query=RecallQuery(text="what happened with the tool")),
        (),
        limit=10,
    )

    # The remembered memory is the closer match; the observed one only outranks it by
    # collecting fusion credit twice for the single cosine its shared vector produced.
    assert [memory.memory_id for memory in fused] == [
        remembered.memory_id,
        observed.memory_id,
    ]


async def test_an_observed_memory_still_reaches_the_memory_record_channel() -> None:
    """The dead channel stays fixed: the ID lookup still receives the derived memory."""
    observed, embeddings, relations = await _observed_memory()
    store = FakeMemoryStore((observed,), relations)
    recall = _recall(store, embeddings)

    fused = await recall._search_semantic_memories(
        RecallRequest(tenant_id=TENANT, query=RecallQuery(text="what happened with the tool")),
        (),
        limit=10,
    )

    assert store.requested_memory_ids == [(observed.memory_id,)]
    assert [memory.memory_id for memory in fused] == [observed.memory_id]


async def test_two_channels_reading_two_vectors_still_add_up() -> None:
    """Removing the duplicate must not cost a memory the channels that are real evidence.

    This memory is found by its own summary and, separately, by the media vector of the clip
    its evidence span was cut from. Those are two measurements of two different objects, which
    is what reciprocal rank fusion exists to add together, so it outranks a closer memory that
    only one channel ever saw.
    """
    observed, embeddings, relations = await _observed_memory()
    remembered = _remembered_memory()
    store = FakeMemoryStore((observed, remembered), relations)
    recall = _recall(
        store,
        (
            *embeddings,
            _remembered_embedding(remembered),
            _evidence_embedding(observed.evidence_ids[0]),
        ),
    )

    fused = await recall._search_semantic_memories(
        RecallRequest(tenant_id=TENANT, query=RecallQuery(text="what happened with the tool")),
        (),
        limit=10,
    )

    assert [memory.memory_id for memory in fused] == [
        observed.memory_id,
        remembered.memory_id,
    ]


async def test_an_entity_match_keeps_promoting_the_memory_it_reached() -> None:
    """A vector no other channel saw must keep moving recall, whatever else shares its row.

    Two observed events, one of them naming the person the query asks about. That one is the
    weaker summary match of the two, and the graph query returns one row per memory however
    many paths reached it, so the entity vector leaves exactly one trace: the memory it
    reached arrives at the front of the graph ranking. Drop that row as a duplicate of the
    summary vector and searching entities stops changing recall at all.
    """
    memories, embeddings, relations = await _observed_memories(
        (_PLACED_TOOL_BY_MARA, _OPENED_TOOLBOX),
        {
            _OPENED_TOOLBOX.description: NEIGHBOUR_VECTOR,
            # The encoder is handed the casefolded name the entity record stores.
            "mara": ENTITY_VECTOR,
        },
    )
    named, neighbour = memories
    remembered = _remembered_memory()
    store = FakeMemoryStore((*memories, remembered), relations)
    recall = _recall(store, (*embeddings, _remembered_embedding(remembered)))

    fused = await recall._search_semantic_memories(
        RecallRequest(tenant_id=TENANT, query=RecallQuery(text="what did mara do")),
        (),
        limit=10,
    )

    # The remembered memory is the closest vector of the three. The named memory outranks the
    # nearer observed one only through the entity, which no other channel searched.
    assert [memory.memory_id for memory in fused] == [
        remembered.memory_id,
        named.memory_id,
        neighbour.memory_id,
    ]


def _recall(store: FakeMemoryStore, embeddings: tuple[EmbeddingRecord, ...]) -> RecallMemories:
    return RecallMemories(
        cast(MemoryStore, store),
        cast(Answerer, None),
        cast(OccurrenceVerifier, None),
        embedding_index=cast(EmbeddingIndex, FakeEmbeddingIndex(embeddings)),
        media_url_signer=cast(MediaUrlSigner, None),
        embedder=cast(Embedder, FixedEmbedder(QUERY_VECTOR)),
        minimum_embedding_similarity=0.0,
    )


async def _observed_memory() -> tuple[
    MemoryRecord, tuple[EmbeddingRecord, ...], tuple[Relation, ...]
]:
    """Derive and index one observed event exactly as the write path does."""
    memories, embeddings, relations = await _observed_memories((_PLACED_TOOL,))
    return memories[0], embeddings, relations


async def _observed_memories(
    perceived_events: tuple[PerceivedEvent, ...],
    by_text: dict[str, tuple[float, ...]] | None = None,
) -> tuple[tuple[MemoryRecord, ...], tuple[EmbeddingRecord, ...], tuple[Relation, ...]]:
    """Derive and index observed events exactly as the write path does."""
    observation = Observation(
        observation_id=ObservationId("observation_01"),
        tenant_id=TENANT,
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_object_ids=(MediaObjectId("media_01"),),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=10),
        observed_at=NOW,
    )
    perception = EventPerception(
        events=perceived_events,
        model_reference=MODEL,
        prompt_version="perceive_events_v4",
    )
    events = tuple(
        Event(
            event_id=EventId(f"event_{index:02d}"),
            tenant_id=TENANT,
            observation_ids=(observation.observation_id,),
            evidence_ids=perceived.evidence_ids,
            occurred_at=observation.occurred_at + timedelta(milliseconds=perceived.start_ms),
            ended_at=observation.occurred_at + timedelta(milliseconds=perceived.end_ms),
            description=perceived.description,
            salience=perceived.salience,
            created_at=NOW,
            model_reference=MODEL,
            prompt_version="perceive_events_v4",
        )
        for index, perceived in enumerate(perceived_events, start=1)
    )
    graph = derive_observation_graph(observation, perception, events, (), NOW)
    embeddings = await embed_observation_graph(
        TENANT,
        events,
        graph.claims,
        graph.entities,
        graph.memories,
        cast(Embedder, FixedEmbedder(DERIVED_VECTOR, by_text)),
        NOW,
    )
    return graph.memories, embeddings, graph.relations


def _remembered_memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId("memory_remembered"),
        tenant_id=TENANT,
        memory_type=MemoryType.SEMANTIC,
        summary="The red tool belongs beside the blue toolbox.",
        evidence_ids=(EvidenceId("evidence_02"),),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.ATTESTED,
    )


def _evidence_embedding(evidence_id: EvidenceId) -> EmbeddingRecord:
    """The clip that span was cut from, encoded as media rather than as its summary."""
    return EmbeddingRecord(
        embedding_id=EmbeddingId("embedding_evidence"),
        tenant_id=TENANT,
        object_type=EmbeddedObjectType.EVIDENCE_SPAN,
        object_id=evidence_id,
        values=EVIDENCE_VECTOR,
        model_reference=EMBEDDING_MODEL,
        space_reference=SPACE,
        task=EmbedTask.DOCUMENT.value,
        dimension=len(EVIDENCE_VECTOR),
        normalized=True,
        created_at=NOW,
    )


def _remembered_embedding(memory: MemoryRecord) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId("embedding_remembered"),
        tenant_id=TENANT,
        object_type=EmbeddedObjectType.MEMORY_RECORD,
        object_id=memory.memory_id,
        values=REMEMBERED_VECTOR,
        model_reference=EMBEDDING_MODEL,
        space_reference=SPACE,
        task=EmbedTask.DOCUMENT.value,
        dimension=len(REMEMBERED_VECTOR),
        normalized=True,
        created_at=NOW,
    )
