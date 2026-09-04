from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
    SpatialAnchor,
    SpatialContext,
)
from mindbridge._telemetry import FORMATION_PROPOSALS_REFUSED


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


def test_a_proposal_the_kernel_refuses_is_dropped_and_not_charged_to_the_write(
    tmp_path: Path,
) -> None:
    """A per-proposal rule costs that proposal, never the observation it was derived from.

    `add` commits the source before formation runs, so raising failed a write that had in fact
    succeeded -- and the record stayed durable with its formation stuck in the queue, so every
    retry re-ran the model and failed the same way. Text that merely mentions an image ("shared a
    photo described as ...") is enough to make a model call it an image cue, which made this the
    ordinary case rather than a rare one. The model adapter already drops and counts a malformed
    proposal; the kernel's own rules now agree with that policy.
    """

    class MisgroundedAffectFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content="The user's preferred drink is tea",
                        subject="user",
                        predicate="preferred_drink",
                        value="tea",
                        confidence=0.9,
                    ),
                    # The source is text; no image was ever observed.
                    FormationProposal(
                        kind=MemoryKind.AFFECT,
                        content="The user looked happy",
                        subject="user",
                        value="happy",
                        cue_modality=Modality.IMAGE,
                    ),
                )
                for _value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=MisgroundedAffectFormer(),
        minimum_relevance=0,
        tracer=provider.get_tracer("test"),
    ) as memory:
        source = memory.add("I prefer tea, and I shared a photo described as a smile")

        assert [item.id for item in memory.list().items] != []
        kinds = {
            hit.context.kind
            for hit in memory.search("preferred drink happy", limit=10)
            if hit.context is not None
        }
        assert MemoryKind.STATE in kinds
        assert MemoryKind.AFFECT not in kinds
        assert memory.pending_captures(memory_ids=(source.id,)) == ()

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


def test_a_damaged_formation_envelope_still_fails_the_write(tmp_path: Path) -> None:
    """Dropping one bad proposal must not swallow a backend that answered the wrong question."""

    class ShortBatchFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert inputs
            return ()

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=ShortBatchFormer(),
        minimum_relevance=0,
    ) as memory:
        with pytest.raises(ModelError) as failure:
            memory.add("I prefer tea")

        assert failure.value.reason == "response_invalid"


def _formed(
    memory: Memory, source_id: str, *, scope: RetrievalScope | None = None
) -> list[SearchHit]:
    hits = memory.search("preferred drink user", limit=10, scope=scope)
    return [hit for hit in hits if hit.id != source_id]


def _entities(memory: Memory, *, scope: RetrievalScope | None = None) -> list[SearchHit]:
    hits = memory.search("entity user", limit=10, scope=scope)
    return [
        hit for hit in hits if hit.context is not None and hit.context.kind is MemoryKind.ENTITY
    ]


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


def test_a_shared_formed_record_keeps_only_what_its_sources_agree_on(tmp_path: Path) -> None:
    """An ENTITY is written once and later sources only add evidence to it.

    Its ID deliberately excludes the source, so the first observation's place and metadata would
    otherwise stand for every later source, including one that disagrees -- filing knowledge in a
    room half its evidence was never observed in. Agreement is the same condition consolidation
    applies to several sources at once.
    """
    with Memory(
        tmp_path / "disagree",
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add(
            "I prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )
        memory.add(
            "I still prefer tea",
            context=ObservationContext(place_id="garage"),
            metadata={"household": "flat-9"},
        )

        (entity,) = _entities(memory)
        assert memory.get(entity.id).place_id is None
        assert memory.get(entity.id).metadata == {}
        assert _entities(memory, scope=RetrievalScope(place_id="kitchen")) == []
        assert _entities(memory, scope=RetrievalScope(place_id="garage")) == []

    with Memory(
        tmp_path / "agree",
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add(
            "I prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )
        memory.add(
            "I still prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )

        (entity,) = _entities(memory, scope=RetrievalScope(place_id="kitchen"))
        assert memory.get(entity.id).place_id == "kitchen"
        assert memory.get(entity.id).metadata == {"household": "flat-2"}


def test_two_contradictory_states_from_one_source_cost_only_the_later_one(
    tmp_path: Path,
) -> None:
    """Two proposals that contradict each other are one model response's grounding fault.

    Both come from a single source the caller already committed, so failing the write reports an
    observation as unwritten that is in fact stored, and every retry re-runs the model and fails
    the same way. The first claim stands, the contradicting one is refused.
    """

    class ContradictingFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                tuple(
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content=f"The user's preferred drink is {drink}",
                        subject="user",
                        predicate="preferred_drink",
                        value=drink,
                        confidence=0.9,
                    )
                    for drink in ("tea", "coffee")
                )
                for _value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=ContradictingFormer(),
        minimum_relevance=0,
        tracer=provider.get_tracer("test"),
    ) as memory:
        source = memory.add("I prefer tea")

        states = [
            hit.context.value
            for hit in _formed(memory, source.id)
            if hit.context is not None and hit.context.kind is MemoryKind.STATE
        ]
        assert states == ["tea"]
        assert memory.pending_captures(memory_ids=(source.id,)) == ()

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


def test_refused_proposals_are_totalled_over_the_whole_settle(tmp_path: Path) -> None:
    """`settle` forms one record at a time, and the operation reports the whole batch.

    Publishing per pass let the last record settled overwrite the count, so a batch whose first
    record lost a proposal reported zero refusals -- the reading an operator would take as proof
    that nothing was lost.
    """

    class PhotoAffectFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=MemoryKind.AFFECT,
                        content="The user looked happy",
                        subject="user",
                        value="happy",
                        # The source is text; no image was ever observed.
                        cue_modality=Modality.IMAGE,
                    ),
                )
                if "photo" in (value.content.text or "")
                else ()
                for value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PhotoAffectFormer(),
        minimum_relevance=0,
        tracer=provider.get_tracer("test"),
    ) as memory:
        memory.capture("I shared a photo described as a smile")
        memory.capture("I prefer tea")

        assert memory.settle() == 2

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.name == "mindbridge.settle"
        and span.attributes is not None
        and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


def test_a_pose_in_another_frame_is_refused_on_the_formation_path(tmp_path: Path) -> None:
    """The spatial rule refuses like the affect rule: the proposal is lost, the write is not.

    A pose the kernel cannot place -- another coordinate frame, another anchor, or a source that
    was never localized -- would otherwise answer a metric scope with a position nothing observed.
    """

    class MisplacedPoseFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=MemoryKind.EVENT,
                        content="The user poured tea in the study",
                        subject="user",
                        spatial=SpatialContext(
                            frame_id="study",
                            anchor=SpatialAnchor.OBSERVER,
                            x=1.0,
                            y=2.0,
                        ),
                        confidence=0.8,
                    ),
                )
                for _value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=MisplacedPoseFormer(),
        minimum_relevance=0,
        tracer=provider.get_tracer("test"),
    ) as memory:
        source = memory.add(
            "I poured tea",
            context=ObservationContext(
                spatial=SpatialContext(
                    frame_id="kitchen", anchor=SpatialAnchor.OBSERVER, x=0.0, y=0.0
                )
            ),
        )

        assert memory.get(source.id).content == "I poured tea"
        assert _formed(memory, source.id) == []
        assert memory.pending_captures(memory_ids=(source.id,)) == ()

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]
