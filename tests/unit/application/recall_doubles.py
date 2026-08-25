"""Store, index and answerer doubles for exercising `RecallMemories` end to end.

Shared by the similarity-floor and answer-round checks, which both need the whole recall path
rather than one private method: what they are about is the interaction between the retrieval
channels, the answer rounds, and what `RecallResult` finally says.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    EmbeddingMatch,
    EmbeddingSearch,
    GeneratedAnswer,
    PresignedMediaDownload,
    ResolvedQueryMedia,
)
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    EmbeddedObjectType,
    EmbeddingSpaceReference,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelReference,
    ObservationId,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
TENANT = TenantId("tenant_recall")
SPACE = EmbeddingSpaceReference(space_id="jina-v5")
EMBEDDING_MODEL = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
QUERY_VECTOR = (1.0, 0.0)


def memory(name: str, summary: str, *, minutes: int = 0) -> MemoryRecord:
    """One episodic memory grounded by an evidence span named after it."""
    return MemoryRecord(
        memory_id=MemoryId(name),
        tenant_id=TENANT,
        memory_type=MemoryType.EPISODIC,
        summary=summary,
        evidence_ids=(EvidenceId(f"evidence_{name}"),),
        occurred_at=NOW + timedelta(minutes=minutes),
        ended_at=NOW + timedelta(minutes=minutes),
        created_at=NOW,
        verification_status=VerificationStatus.VERIFIED,
    )


def unit_vector(cosine: float) -> tuple[float, float]:
    """A unit vector whose cosine against `QUERY_VECTOR` is exactly `cosine`."""
    return (cosine, (1.0 - cosine * cosine) ** 0.5)


class FixedEmbedder:
    """Encode every query to the same vector; similarity lives in the index double."""

    model_reference = EMBEDDING_MODEL
    space_reference = SPACE

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        return EmbedResult(
            tuple(
                Embedding(QUERY_VECTOR, self.model_reference, self.space_reference)
                for _ in request.inputs
            )
        )


class SimilarityIndex:
    """Rank memory-record vectors by cosine and apply the floor pgvector applies.

    `_search_semantic_memories` issues exactly three searches per retrieval wave, so a caller that
    wants a follow-up wave to match something else passes a second mapping and it takes effect
    from the fourth search on. The last mapping given serves every wave after it.
    """

    _SEARCHES_PER_WAVE = 3

    def __init__(self, *waves: dict[str, float]) -> None:
        if not waves:
            raise ValueError("at least one wave of similarities is required")
        self.waves = waves
        self.searches: list[EmbeddingSearch] = []

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        similarities = self.waves[min(self.wave_index, len(self.waves) - 1)]
        self.searches.append(search)
        if EmbeddedObjectType.MEMORY_RECORD not in search.object_types:
            return ()
        ranked = sorted(similarities.items(), key=lambda item: -item[1])
        return tuple(
            EmbeddingMatch(
                embedding_id=f"embedding_{memory_id}",
                object_type=EmbeddedObjectType.MEMORY_RECORD,
                object_id=memory_id,
                similarity=similarity,
            )
            for memory_id, similarity in ranked
            if similarity >= search.minimum_similarity
        )[: search.limit]

    @property
    def wave_index(self) -> int:
        """Which retrieval wave the next search belongs to, counting from zero."""
        return len(self.searches) // self._SEARCHES_PER_WAVE


class RecallStore:
    """The persistence side of recall, answering each channel the way PostgreSQL does."""

    def __init__(
        self,
        memories: tuple[MemoryRecord, ...],
        *,
        lexical: frozenset[str] = frozenset(),
    ) -> None:
        self.memories = {item.memory_id: item for item in memories}
        # Which memory IDs the full-text channel matches. That channel ORs the query's lexemes
        # and has no similarity floor of its own, so what it admits is a property of the corpus,
        # not of the configured floor -- which is exactly what these tests are about.
        self.lexical = lexical
        self.evidence_reads: list[tuple[EvidenceId, ...]] = []
        self.presigns: list[MediaObjectId] = []

    async def search_memories(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(self.memories[MemoryId(name)] for name in sorted(self.lexical))[:limit]

    async def search_memories_by_ids(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            self.memories[memory_id]
            for memory_id in ranked_memory_ids
            if memory_id in self.memories
        )[:limit]

    async def search_memories_by_evidence(
        self,
        request: RecallRequest,
        ranked_evidence_ids: tuple[EvidenceId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return ()

    async def search_memories_by_hierarchy(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return ()

    async def search_memories_by_graph_objects(
        self,
        request: RecallRequest,
        ranked_objects: tuple[EmbeddingMatch, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return ()

    async def record_memory_accesses(
        self,
        tenant_id: TenantId,
        memory_ids: tuple[MemoryId, ...],
        *,
        accessed_at: datetime,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(self.memories[memory_id] for memory_id in memory_ids)

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        self.evidence_reads.append(evidence_ids)
        return tuple(
            EvidenceSpan(
                evidence_id=evidence_id,
                tenant_id=TENANT,
                observation_id=ObservationId(f"observation_{evidence_id}"),
                media_object_id=MediaObjectId(f"media_{evidence_id}"),
                start_ms=0,
                end_ms=4_000,
                created_at=NOW,
            )
            for evidence_id in evidence_ids
        )

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]:
        return tuple(
            MediaObject(
                media_object_id=media_object_id,
                tenant_id=TENANT,
                kind=MediaKind.VIDEO,
                uri=f"s3://media/{TENANT}/{media_object_id}.mp4",
                sha256=f"{abs(hash(media_object_id)):064x}"[:64],
                size_bytes=1_000,
                created_at=NOW,
                duration_ms=4_000,
            )
            for media_object_id in media_object_ids
        )

    async def read_evidence_clip_media(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> dict[EvidenceId, MediaObject]:
        return {}

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        self.presigns.append(media_object.media_object_id)
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=35),
        )


class ScriptedAnswerer:
    """Return a queued answer per round, or summarise the first candidate."""

    def __init__(self, answers: tuple[GeneratedAnswer, ...] = ()) -> None:
        self.answers = list(answers)
        self.rounds: list[tuple[MemoryRecord, ...]] = []
        self.evidence_rounds: list[tuple[ResolvedEvidence, ...]] = []

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer:
        self.rounds.append(memories)
        self.evidence_rounds.append(evidence)
        if self.answers:
            return self.answers.pop(0)
        if not memories:
            return GeneratedAnswer(answer=None, confidence=0.0)
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)
