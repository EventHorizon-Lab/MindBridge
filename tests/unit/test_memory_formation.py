from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    Blob,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    Modality,
    ModelError,
    ObservationContext,
    RetrievalScope,
    SearchHit,
)


class PreferenceFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "preference-test"
    formation_space = "preference-test:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        return tuple(
            (
                FormationProposal(
                    kind=MemoryKind.ENTITY,
                    content="The user is an entity",
                    subject="user",
                    confidence=0.99,
                ),
                FormationProposal(
                    kind=MemoryKind.STATE,
                    content="The user's preferred drink is tea",
                    subject="user",
                    predicate="preferred_drink",
                    value="tea",
                    confidence=0.9,
                    valid_from=value.context.valid_from,
                ),
            )
            for value in inputs
        )

    def close(self) -> None:
        pass


def test_new_observation_forms_typed_memories_once_without_changing_source_identity(
    tmp_path: Path,
) -> None:
    occurred = datetime(2026, 1, 2, tzinfo=timezone.utc)
    context = ObservationContext(
        basis=EvidenceBasis.USER_STATEMENT,
        source_id="turn-17",
        valid_from=occurred,
    )
    former = PreferenceFormer()

    with Memory(
        tmp_path / "formed",
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
    ) as memory:
        source = memory.add("I prefer tea", occurred_at=occurred, context=context)
        duplicate = memory.add("I prefer tea", occurred_at=occurred, context=context)
        hits = memory.search("preferred drink", limit=10)

        assert source == duplicate
        assert former.calls == 1
        assert source.context is not None
        assert source.context.kind is MemoryKind.OBSERVATION
        states = [hit for hit in hits if hit.context and hit.context.kind is MemoryKind.STATE]
        assert len(states) == 1
        state = states[0].context
        assert state is not None
        assert state.subject == "user"
        assert state.predicate == "preferred_drink"
        assert state.value == "tea"
        assert state.evidence_ids == (source.id,)
        assert state.model_id == "preference-test"

    with Memory(tmp_path / "plain", embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        plain = memory.add("I prefer tea", occurred_at=occurred, context=context)

    assert plain.id == source.id


def test_concurrent_duplicate_source_runs_formation_once(tmp_path: Path) -> None:
    occurred = datetime(2026, 1, 2, tzinfo=timezone.utc)
    context = ObservationContext(source_id="same-turn", valid_from=occurred)
    former = PreferenceFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
    ) as memory:
        with ThreadPoolExecutor(max_workers=2) as executor:
            records = tuple(
                executor.map(
                    lambda _index: memory.add(
                        "I prefer tea",
                        occurred_at=occurred,
                        context=context,
                    ),
                    range(2),
                )
            )

        assert records[0] == records[1]
        assert former.calls == 1


def test_implicit_and_explicit_default_context_have_one_identity(tmp_path: Path) -> None:
    former = PreferenceFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
    ) as memory:
        implicit = memory.add("I prefer tea")
        explicit = memory.add("I prefer tea", context=ObservationContext())

        assert implicit == explicit
        assert former.calls == 1

        implicit_batch = memory.add_many(("batch tea one", "batch tea two"))
        explicit_batch = memory.add_many(
            ("batch tea one", "batch tea two"),
            context=(ObservationContext(), ObservationContext()),
        )
        assert implicit_batch == explicit_batch
        assert former.calls == 2


def test_known_at_only_exposes_evidence_known_at_that_time(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        first = memory.add(
            "first entity witness",
            context=ObservationContext(source_id="turn-1"),
        )
        known_after_first = datetime.now(timezone.utc)
        second = memory.add(
            "second entity witness",
            context=ObservationContext(source_id="turn-2"),
        )

        historical = next(
            hit.context
            for hit in memory.search(
                "user entity",
                limit=10,
                scope=RetrievalScope(known_at=known_after_first),
            )
            if hit.context is not None and hit.context.kind is MemoryKind.ENTITY
        )
        current = next(
            hit.context
            for hit in memory.search("user entity", limit=10)
            if hit.context is not None and hit.context.kind is MemoryKind.ENTITY
        )

        assert historical.evidence_ids == (first.id,)
        assert current.evidence_ids == (first.id, second.id)


def test_deleting_a_source_removes_unsupported_derived_records(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        source = memory.add("I prefer tea")
        assert any(hit.context is not None for hit in memory.search("preferred drink", limit=10))

        assert memory.delete(source.id) is True
        assert memory.delete(source.id) is False
        assert memory.search("preferred drink", limit=10) == ()


def test_unsupported_source_modality_is_kept_without_calling_the_former(tmp_path: Path) -> None:
    former = PreferenceFormer()
    former.formation_capabilities = frozenset({Modality.TEXT})
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
    ) as memory:
        first = memory.add(Blob(b"audio", "audio/wav"))
        duplicate = memory.add(Blob(b"audio", "audio/wav"))

        assert first == duplicate
        assert former.calls == 0
        assert first.context is not None
        assert first.context.kind is MemoryKind.OBSERVATION

    former.formation_capabilities = ATOMIC_MODALITIES
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
    ) as memory:
        upgraded = memory.add(Blob(b"audio", "audio/wav"))
        assert upgraded == first
        assert former.calls == 1


def test_affect_proposal_must_name_a_modality_present_in_its_source(tmp_path: Path) -> None:
    class InvalidAffectFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=MemoryKind.AFFECT,
                        content="The user sounded happy",
                        subject="user",
                        value="happy",
                        cue_modality=Modality.AUDIO,
                    ),
                )
                for _value in inputs
            )

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InvalidAffectFormer(),
        minimum_relevance=0,
    ) as memory:
        with pytest.raises(ModelError, match="modality present"):
            memory.add("I am happy")

        assert [item.content for item in memory.list().items] == ["I am happy"]


def _formed(
    memory: Memory, source_id: str, *, scope: RetrievalScope | None = None
) -> list[SearchHit]:
    hits = memory.search("preferred drink user", limit=10, scope=scope)
    return [hit for hit in hits if hit.id != source_id]


def test_formed_records_inherit_the_place_they_were_observed_in(tmp_path: Path) -> None:
    """ "What do we know about the kitchen?" must see the knowledge, not only the raw observation.

    `place_id` is a hard filter on the record, so a formed record that does not carry the place of
    the observation it was formed from is unreachable from every place-scoped `search`, `ask`, and
    `compile` -- the household question the symbolic axis exists for.
    """
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        source = memory.add("I prefer tea", context=ObservationContext(place_id="kitchen"))

        kitchen = _formed(memory, source.id, scope=RetrievalScope(place_id="kitchen"))
        garage = _formed(memory, source.id, scope=RetrievalScope(place_id="garage"))

        assert {hit.context.kind for hit in kitchen if hit.context} == {
            MemoryKind.ENTITY,
            MemoryKind.STATE,
        }
        assert {hit.place_id for hit in kitchen} == {"kitchen"}
        assert garage == []
        assert {memory.get(hit.id).place_id for hit in kitchen} == {"kitchen"}


def test_formed_records_inherit_the_metadata_of_their_source(tmp_path: Path) -> None:
    """A host that filters recall by metadata expects formed knowledge to carry the same tag."""
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        source = memory.add("I prefer tea", metadata={"household": "flat-2"})

        formed = _formed(memory, source.id)

        assert formed
        assert all(hit.metadata == {"household": "flat-2"} for hit in formed)


def test_a_formed_record_from_a_placeless_source_has_no_place(tmp_path: Path) -> None:
    """Inheriting a place must not invent one: an unlabelled observation is not "everywhere"."""
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        source = memory.add("I prefer tea")

        formed = _formed(memory, source.id)

        assert formed
        assert all(hit.place_id is None for hit in formed)
        assert _formed(memory, source.id, scope=RetrievalScope(place_id="kitchen")) == []
