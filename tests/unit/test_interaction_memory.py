from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    Blob,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    Modality,
    ObservationContext,
    RetrievalScope,
)


class InteractionFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "interaction-test"
    formation_space = "interaction-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        results: list[tuple[FormationProposal, ...]] = []
        for item in inputs:
            if "trait" in item.content.text:
                style = "detailed" if "detailed" in item.content.text else "concise"
                results.append(
                    (
                        FormationProposal(
                            kind=MemoryKind.TRAIT,
                            content=f"The user tends to prefer {style} responses",
                            basis=(
                                item.context.basis
                                if item.content.text.startswith("trait:")
                                else EvidenceBasis.MODEL_INFERENCE
                            ),
                            subject="user",
                            predicate="response_style",
                            value=style,
                            confidence=0.7,
                        ),
                    )
                )
                continue
            results.append(
                (
                    FormationProposal(
                        kind=MemoryKind.AFFECT,
                        content="Text affect cue: calm",
                        subject="user",
                        value="calm",
                        cue_modality=Modality.TEXT,
                        valence=0.4,
                        arousal=0.2,
                        confidence=0.8,
                    ),
                    FormationProposal(
                        kind=MemoryKind.AFFECT,
                        content="Audio affect cue: tense",
                        subject="user",
                        value="tense",
                        cue_modality=Modality.AUDIO,
                        valence=-0.2,
                        arousal=0.8,
                        confidence=0.65,
                    ),
                )
            )
        return tuple(results)

    def close(self) -> None:
        pass


def test_multimodal_affect_keeps_conflicting_cues_separate(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add(("I am fine", Blob(b"voice", "audio/wav")))
        affect = [
            hit.context
            for hit in memory.search("affect cue", limit=10)
            if hit.context is not None and hit.context.kind is MemoryKind.AFFECT
        ]

        assert {value.cue_modality for value in affect} == {
            Modality.TEXT,
            Modality.AUDIO,
        }
        assert {value.value for value in affect} == {"calm", "tense"}


def test_a_trait_needs_a_statement_or_two_independent_observations(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("trait evidence one")
        assert not any(
            hit.context and hit.context.kind is MemoryKind.TRAIT
            for hit in memory.search("concise responses", limit=10)
        )

        memory.add("trait evidence two")
        traits = [
            hit.context
            for hit in memory.search("concise responses", limit=10)
            if hit.context and hit.context.kind is MemoryKind.TRAIT
        ]
        assert len(traits) == 1
        assert len(traits[0].evidence_ids) == 2

    with Memory(
        tmp_path / "statement",
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add(
            "trait: I prefer concise responses",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        assert any(
            hit.context and hit.context.kind is MemoryKind.TRAIT
            for hit in memory.search("concise responses", limit=10)
        )


def test_trait_support_counts_independent_sources_and_retracts_on_delete(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        first = memory.add(
            "trait evidence one",
            context=ObservationContext(source_id="same-session"),
        )
        memory.add(
            "trait evidence duplicate",
            context=ObservationContext(source_id="same-session"),
        )
        assert not any(
            hit.context and hit.context.kind is MemoryKind.TRAIT
            for hit in memory.search("concise responses", limit=10)
        )

        independent = memory.add(
            "trait evidence independent",
            context=ObservationContext(source_id="other-session"),
        )
        assert any(
            hit.context and hit.context.kind is MemoryKind.TRAIT
            for hit in memory.search("concise responses", limit=10)
        )
        known_before_delete = datetime.now(timezone.utc)

        assert memory.delete(independent.id) is True
        assert not any(
            hit.context and hit.context.kind is MemoryKind.TRAIT
            for hit in memory.search("concise responses", limit=10)
        )
        historical = next(
            hit.context
            for hit in memory.search(
                "concise responses",
                limit=10,
                scope=RetrievalScope(known_at=known_before_delete),
            )
            if hit.context and hit.context.kind is MemoryKind.TRAIT
        )
        assert independent.id in historical.evidence_ids
        assert memory.get(first.id) == first


def test_explicit_typed_trait_statement_supersedes_inferred_trait(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("trait evidence one")
        memory.add("trait evidence two")
        memory.add(
            "trait: I prefer detailed responses",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )

        values = {
            hit.context.value
            for hit in memory.search("responses", limit=10)
            if hit.context and hit.context.kind is MemoryKind.TRAIT
        }

        assert values == {"detailed"}


def test_explicit_trait_precedence_is_insertion_order_independent(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add(
            "trait: I prefer concise responses",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        memory.add("trait detailed evidence one")
        memory.add("trait detailed evidence two")

        traits = {
            (hit.context.value, hit.context.basis)
            for hit in memory.search("responses", limit=10)
            if hit.context and hit.context.kind is MemoryKind.TRAIT
        }

    assert traits == {("concise", EvidenceBasis.USER_STATEMENT)}


def test_explicit_trait_can_evolve_back_to_an_earlier_value(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    april = datetime(2026, 4, 1, tzinfo=timezone.utc)
    statement = ObservationContext(basis=EvidenceBasis.USER_STATEMENT)

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InteractionFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("trait: concise", occurred_at=january, context=statement)
        memory.add("trait: detailed", occurred_at=february, context=statement)
        memory.add("trait: concise", occurred_at=march, context=statement)

        traits = {
            hit.context.value
            for hit in memory.search(
                "responses",
                limit=10,
                scope=RetrievalScope(valid_at=april),
            )
            if hit.context and hit.context.kind is MemoryKind.TRAIT
        }

    assert traits == {"concise"}
