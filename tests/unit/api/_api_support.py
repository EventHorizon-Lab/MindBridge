"""Shared values and result fixtures for REST and MCP adapter tests."""

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from mindbridge.types import (
    AffectCue,
    AssetRef,
    ContextBudget,
    ContextBundle,
    ContextConflict,
    ContextUnknown,
    ContextUnknownKind,
    EvidenceBasis,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryContext,
    MemoryKind,
    MemoryRecord,
    MemoryType,
    Modality,
    NamedActor,
    ProvisionalActor,
    RetrievalCandidateTrace,
    RetrievalRejection,
    RetrievalTrace,
    SearchHit,
    SpeakerSegment,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
OCCURRED_FROM = datetime(2026, 8, 27, tzinfo=timezone.utc)
OCCURRED_UNTIL = datetime(2026, 8, 28, tzinfo=timezone.utc)
ENVELOPE_FIELDS = {
    "code",
    "reason",
    "retryable",
    "stage",
    "subject",
    "message",
    "trace_id",
    "issues",
}
ASSET = AssetRef(
    id="asset_image",
    modality=Modality.IMAGE,
    media_type="image/png",
    size_bytes=3,
    sha256="a" * 64,
    name="frame.png",
    path=Path("/private/mindbridge/assets/frame.png"),
)
PROVISIONAL = ProvisionalActor(identity_id="identity_2", memory_ids=("memory_2",))
AFFECT_CONTEXT = MemoryContext(
    kind=MemoryKind.AFFECT,
    basis=EvidenceBasis.MODEL_INFERENCE,
    confidence=0.72,
    valid_from=None,
    valid_until=None,
    recorded_at=NOW,
    evidence_ids=("memory_2",),
    cue_modality=Modality.AUDIO,
    valence=-0.2,
    arousal=0.8,
)
NAMED = NamedActor(
    identity_id="identity_3",
    name="Alice",
    memory_ids=("memory_2",),
    naming_assertion_id="memory_3",
)
CONFLICT = ContextConflict(
    lineage_id="lineage_1",
    subject="ana",
    predicate="location",
    values=("berlin", "paris"),
    memory_ids=("memory_1", "memory_2"),
)
UNKNOWN = ContextUnknown(
    kind=ContextUnknownKind.BUDGET_EXCLUDED,
    detail="3 candidates did not fit 24 items and 16000 chars",
)
EVIDENCE_UNAVAILABLE = ContextUnknown(
    kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
    detail="a required supporting memory was unavailable",
)
SEGMENT = SpeakerSegment(
    asset_id="b" * 64,
    start_ms=0,
    end_ms=1500,
    text="Where is the toolbox?",
    speaker_id="speaker_1",
    speaker_name="Ann",
    identity_score=0.82,
)
FACE = FaceObservation(
    asset_id="c" * 64,
    bounding_box=(0.1, 0.2, 0.3, 0.4),
    identity_id="identity_1",
    identity_name="Ann",
    identity_score=0.77,
    observed_at_ms=250,
)
ERASURE = IdentityErasure(
    identity_id="identity_1",
    alias_ids=("identity_4",),
    face_exemplars=3,
    voice_exemplars=2,
    face_observations=7,
    speech_segments=11,
)
PROFILE = IdentityProfile(
    identity_id="identity_1",
    name="Ann",
    relationship="daughter",
    confirmed=True,
    evidence_ids=("memory_1",),
)


def _record(
    memory_id: str = "memory_1",
    content: str = "The toolbox is blue.",
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    assets: tuple[AssetRef, ...] = (),
    modality: Modality = Modality.TEXT,
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        modality=modality,
        memory_type=memory_type,
        assets=assets,
        created_at=NOW,
        occurred_at=occurred_at,
        occurred_end=occurred_end,
        metadata=metadata or {},
    )


def _bundle(budget: ContextBudget, reference_at: datetime) -> ContextBundle:
    """Two populated sections, empty ones, one conflict, and one explicit unknown."""
    return ContextBundle(
        goal="What should I bring?",
        reference_at=reference_at,
        budget=budget,
        actors=(PROVISIONAL,),
        relationships=(),
        scene=(),
        episodes=(_media_hit(),),
        facts=(_hit(),),
        procedures=(),
        affect=(_affect_cue(),),
        traits=(),
        conflicts=(CONFLICT,),
        unknowns=(UNKNOWN, EVIDENCE_UNAVAILABLE),
        occurred_from=OCCURRED_FROM,
        occurred_until=OCCURRED_UNTIL,
        frames=("home/map",),
        places=("kitchen",),
        omitted=3,
        chars=42,
        elapsed_ms=7,
        deadline_exceeded=False,
    )


def _affect_cue() -> AffectCue:
    """One affect entry, whose evidence hop must reach the wire beside the hit fields."""
    return AffectCue(
        id="memory_3",
        content="The user sounded tense.",
        score=0.6,
        memory_type=MemoryType.EPISODIC,
        created_at=NOW,
        context=AFFECT_CONTEXT,
        event_ids=("memory_9",),
    )


def _media_hit() -> SearchHit:
    """A hit carrying a stored asset, whose local path must never be serialized."""
    return SearchHit(
        id="memory_2",
        content="The toolbox is blue.",
        score=0.7,
        modality=Modality.IMAGE,
        memory_type=MemoryType.EPISODIC,
        assets=(ASSET,),
        created_at=NOW,
    )


def _trace() -> RetrievalTrace:
    return RetrievalTrace(
        candidates=(
            RetrievalCandidateTrace(
                memory_id="memory_1",
                index_ids=("index_1",),
                dense_relevance=0.42,
                gate_relevance=0.31,
                final_score=0.4,
                rejected_by=RetrievalRejection.MINIMUM_RELEVANCE,
            ),
        ),
        candidate_limit=50,
        exhaustive=True,
        ambiguous=True,
    )


def _hit() -> SearchHit:
    return SearchHit(
        id="memory_1",
        content="The toolbox is blue.",
        score=0.9,
        modality=Modality.TEXT,
        created_at=NOW,
    )
