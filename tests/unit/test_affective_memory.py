"""Executable contract for the affective-memory principles in `docs/affective-memory.md`.

Model-free: one scripted `FormationBackend` turns a small companion dialogue into typed
proposals, and every assertion is about public SDK behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from _feature_support import ATOMIC_MODALITIES, CueConsolidator, TinyEmbedder

from mindbridge import (
    Blob,
    ContextBudget,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    MemoryRecord,
    Modality,
    ObservationContext,
)

MISSING_DOG = "the user's voice rose while telling me the dog Max went missing"
CORRECTION = "the user said: I wasn't angry, I was nervous about the missing dog"
SIGH = "the user sighed about the missing dog and said it was fine"
MEDICATION = "the user takes their medication lamotrigine every morning"

DOG_GOAL = "how did the user feel about the missing dog"


class CompanionFormer:
    """Scripted former: each companion turn maps to the proposals that turn licenses."""

    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "companion-test"
    formation_space = "companion-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(self._proposals(item.content.text) for item in inputs)

    def _proposals(self, text: str) -> tuple[FormationProposal, ...]:
        if "went missing" in text:
            return (
                FormationProposal(kind=MemoryKind.EVENT, content="the dog Max went missing"),
                FormationProposal(
                    kind=MemoryKind.AFFECT,
                    content="the user seemed anxious about the missing dog",
                    subject="user",
                    value="anxious",
                    confidence=0.7,
                    cue_modality=Modality.TEXT,
                    valence=-0.6,
                    arousal=0.7,
                ),
            )
        if "wasn't angry" in text:
            return (
                FormationProposal(
                    kind=MemoryKind.AFFECT,
                    content="the user says they were nervous about the missing dog, not angry",
                    basis=EvidenceBasis.USER_STATEMENT,
                    subject="user",
                    value="nervous",
                    cue_modality=Modality.TEXT,
                    valence=-0.3,
                    arousal=0.6,
                ),
            )
        if "sighed" in text:
            return (
                FormationProposal(
                    kind=MemoryKind.AFFECT,
                    content="the words about the missing dog read as settled",
                    subject="user",
                    value="settled",
                    confidence=0.5,
                    cue_modality=Modality.TEXT,
                    valence=0.2,
                    arousal=0.2,
                ),
                FormationProposal(
                    kind=MemoryKind.AFFECT,
                    content="the voice about the missing dog sounded weary",
                    subject="user",
                    value="weary",
                    confidence=0.6,
                    cue_modality=Modality.AUDIO,
                    valence=-0.5,
                    arousal=0.3,
                ),
            )
        return ()

    def close(self) -> None:
        return None


def _memory(data_dir: Path) -> Memory:
    return Memory(
        data_dir,
        embedder=TinyEmbedder(),
        former=CompanionFormer(),
        consolidator=CueConsolidator(),
        minimum_relevance=0,
    )


def _records(memory: Memory, kind: MemoryKind) -> tuple[MemoryRecord, ...]:
    return tuple(
        record
        for record in memory.list(limit=100).items
        if record.context is not None and record.context.kind is kind
    )


def test_a_cue_an_event_and_an_observation_stay_three_traceable_records(tmp_path: Path) -> None:
    """Affective cue, affective state, and emotional event are separate layers, each sourced."""
    with _memory(tmp_path) as memory:
        observation = memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        bundle = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        (cue,) = bundle.affect
        (event,) = _records(memory, MemoryKind.EVENT)
        assert cue.context is not None
        assert cue.context.evidence_ids == (observation.id,)
        assert cue.event_ids == (event.id,)
        assert event.id in {hit.id for hit in bundle.episodes}
        assert cue.context.basis is EvidenceBasis.MODEL_INFERENCE
        line = next(row for row in bundle.render().splitlines() if row.startswith(f"- [{cue.id}]"))
        assert "basis model_inference" in line
        assert "cue text" in line
        assert "valence -0.60" in line
        assert "arousal 0.70" in line


def test_a_user_statement_coexists_with_the_inference_it_corrects(tmp_path: Path) -> None:
    """A statement forms a higher-authority version; it never deletes the inference."""
    with _memory(tmp_path) as memory:
        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        memory.add(CORRECTION, context=ObservationContext(source_id="turn-2"))
        cues = _records(memory, MemoryKind.AFFECT)
        bundle = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        assert {record.context.value for record in cues if record.context} == {
            "anxious",
            "nervous",
        }
        assert all(memory.get(record.id) == record for record in cues)
        assert {hit.id for hit in bundle.affect} == {record.id for record in cues}
        # Affect is outside the conflict kinds, so two disagreeing cues are two records to read,
        # not a contradiction to reconcile.
        assert bundle.conflicts == ()
        statement = next(
            hit for hit in bundle.affect if hit.context and hit.context.value == "nervous"
        )
        assert "basis user_statement" in next(
            row for row in bundle.render().splitlines() if row.startswith(f"- [{statement.id}]")
        )


def test_two_modalities_of_one_moment_stay_two_cues(tmp_path: Path) -> None:
    """Per-modality estimates stay separate; fusion is never stored as one label."""
    with _memory(tmp_path) as memory:
        memory.add(
            (SIGH, Blob(b"sigh", "audio/wav")),
            context=ObservationContext(source_id="turn-3"),
        )
        cues = _records(memory, MemoryKind.AFFECT)
        bundle = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        assert {record.context.cue_modality for record in cues if record.context} == {
            Modality.TEXT,
            Modality.AUDIO,
        }
        assert {record.context.valence for record in cues if record.context} == {0.2, -0.5}
        assert {hit.id for hit in bundle.affect} == {record.id for record in cues}


def test_a_second_independent_capture_reinforces_the_same_trait_rather_than_creating_a_rival(
    tmp_path: Path,
) -> None:
    """The claim is the same, so a second capture reinforces the record instead of rivalling it."""
    with _memory(tmp_path) as memory:
        memory.add(
            (SIGH, Blob(b"sigh", "audio/wav")),
            context=ObservationContext(source_id="turn-3"),
        )
        one_capture = tuple(record.id for record in _records(memory, MemoryKind.AFFECT))
        first = memory.consolidate(evidence_ids=one_capture)
        (trait_id,) = first.operations[0].created_ids

        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        (independent,) = (
            record.id
            for record in _records(memory, MemoryKind.AFFECT)
            if record.id not in one_capture
        )
        second = memory.consolidate(evidence_ids=(one_capture[0], independent))

        assert second.rejected == ()
        assert second.operations[0].changed_ids == (trait_id,)
        assert second.operations[0].created_ids == ()


def test_forgetting_a_cue_is_cognitive_and_reversible(tmp_path: Path) -> None:
    """Affect can be forgotten without erasing the audit trail, and forgetting is reversible."""
    with _memory(tmp_path) as memory:
        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        (cue,) = _records(memory, MemoryKind.AFFECT)

        operation = memory.forget((cue.id,))
        assert operation is not None
        assert memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8)).affect == ()
        assert not any(hit.id == cue.id for hit in memory.search(DOG_GOAL, limit=10))
        assert memory.get(cue.id).forgotten_at is not None

        assert memory.rollback(operation.operation_id) is True
        restored = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        assert [hit.id for hit in restored.affect] == [cue.id]
        assert memory.get(cue.id).forgotten_at is None


def test_affect_is_not_a_retrieval_key_for_a_factual_goal(tmp_path: Path) -> None:
    """Ranking is relevance, person, time, and task; affect never buys a slot it did not earn."""
    with _memory(tmp_path) as memory:
        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        medication = memory.add(MEDICATION, context=ObservationContext(source_id="turn-5"))
        bundle = memory.compile(
            "what medication does the user take",
            budget=ContextBudget(max_items=1),
        )

        assert [hit.id for hit in bundle.hits] == [medication.id]
        assert bundle.affect == ()
