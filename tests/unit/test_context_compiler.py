"""Context compilation: partitioning, budgeting, conflicts, rendering, and advertisement."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from test_memory_control_plane import ScriptedConsolidator

from mindbridge import (
    AnswerResult,
    AssetRef,
    AsyncMemory,
    ContextBudget,
    ContextBundle,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryCapabilities,
    MemoryContext,
    MemoryKind,
    MemoryType,
    Modality,
    ModelInput,
    ObservationContext,
    ProvisionalActor,
    SearchHit,
    SpatialAnchor,
    SpatialContext,
    ValidationError,
)
from mindbridge.cli import _LOCAL, _parser
from mindbridge.context import compile_context, evidence_cost

REFERENCE = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)


def _hit(
    identifier: str,
    *,
    score: float = 0.5,
    content: str | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    kind: MemoryKind | None = None,
    confidence: float = 0.9,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    created_at: datetime = REFERENCE,
    lineage_id: str | None = None,
    value: str | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    frame_id: str | None = None,
    assets: tuple[AssetRef, ...] = (),
    modality: Modality = Modality.TEXT,
) -> SearchHit:
    context = (
        None
        if kind is None
        else MemoryContext(
            kind=kind,
            basis=EvidenceBasis.OBSERVATION,
            confidence=confidence,
            valid_from=valid_from,
            valid_until=valid_until,
            recorded_at=created_at,
            lineage_id=lineage_id,
            subject="ana" if value is not None else None,
            predicate="location" if value is not None else None,
            value=value,
            spatial=(
                None
                if frame_id is None
                else SpatialContext(frame_id=frame_id, anchor=SpatialAnchor.OBSERVER, x=0.0, y=0.0)
            ),
        )
    )
    return SearchHit(
        id=identifier,
        content=content or f"evidence {identifier}",
        score=score,
        created_at=created_at,
        occurred_at=occurred_at,
        occurred_end=occurred_end,
        memory_type=memory_type,
        modality=modality,
        assets=assets,
        context=context,
    )


def _compile(hits: Sequence[SearchHit], **budget: object) -> ContextBundle:
    return compile_context(
        "what is going on",
        hits,
        budget=ContextBudget(**budget),  # type: ignore[arg-type]
        reference_at=REFERENCE,
    )


# ---------------------------------------------------------------------------------------------
# Budget validation


@pytest.mark.parametrize(
    "invalid",
    (
        {"max_chars": 0},
        {"max_chars": True},
        {"max_items": -1},
        {"min_confidence": 1.5},
        {"freshness": timedelta(0)},
        {"freshness": 60},
        {"memory_types": frozenset()},
        {"memory_types": frozenset({"semantic"})},
    ),
)
def test_a_budget_that_cannot_bound_anything_is_rejected(invalid: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ContextBudget(**invalid)  # type: ignore[arg-type]


def test_budget_defaults_are_the_documented_ones() -> None:
    budget = ContextBudget()
    assert (budget.max_chars, budget.max_items) == (16_000, 24)
    assert (budget.memory_types, budget.min_confidence, budget.freshness) == (None, 0.0, None)


def test_the_default_budget_admits_one_video_memory() -> None:
    """An omni-modal product whose default budget cannot hold its primary modality is broken.

    The selector skips a hit that does not fit rather than truncating it, so a ceiling below one
    video part's charge does not shrink video evidence, it removes all of it and reports the
    records as omitted. A caller would have to discover that by tuning.
    """
    video = _hit(
        "video_1",
        content="the neighbour is at the door",
        modality=Modality.VIDEO,
        assets=(
            AssetRef(
                id="asset-video",
                modality=Modality.VIDEO,
                media_type="video/mp4",
                size_bytes=1,
                sha256="a" * 64,
                path=Path("door.mp4"),
            ),
        ),
    )
    assert evidence_cost(video) <= ContextBudget().max_chars

    bundle = compile_context(
        "who is at the door", (video,), budget=ContextBudget(), reference_at=REFERENCE
    )
    assert [hit.id for hit in bundle.hits] == ["video_1"]
    assert bundle.omitted == 0


# ---------------------------------------------------------------------------------------------
# Partitioning and selection


def test_hits_partition_by_memory_type_and_kind() -> None:
    bundle = _compile(
        (
            _hit("actor", kind=MemoryKind.ENTITY, memory_type=MemoryType.SEMANTIC),
            _hit("fact", kind=MemoryKind.STATE, memory_type=MemoryType.SEMANTIC),
            _hit("untyped", memory_type=MemoryType.SEMANTIC),
            _hit("episode", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC),
            _hit("procedure", kind=MemoryKind.RESPONSE_POLICY, memory_type=MemoryType.PROCEDURAL),
            _hit("cue", kind=MemoryKind.AFFECT, memory_type=MemoryType.EPISODIC),
            _hit("trait", kind=MemoryKind.TRAIT, memory_type=MemoryType.SEMANTIC),
        )
    )

    assert [hit.id for hit in bundle.actors if isinstance(hit, SearchHit)] == ["actor"]
    assert [hit.id for hit in bundle.facts] == ["fact", "untyped"]
    assert [hit.id for hit in bundle.episodes] == ["episode"]
    assert [hit.id for hit in bundle.procedures] == ["procedure"]
    assert [hit.id for hit in bundle.affect] == ["cue"]
    assert [hit.id for hit in bundle.traits] == ["trait"]
    assert len(bundle.hits) == 7


def test_every_non_empty_section_is_served_before_any_section_is_served_twice() -> None:
    hits = (
        *(
            _hit(f"fact-{index}", score=0.9 - index / 100, kind=MemoryKind.STATE)
            for index in range(5)
        ),
        _hit("trait-0", score=0.2, kind=MemoryKind.TRAIT),
        _hit("cue-0", score=0.1, kind=MemoryKind.AFFECT),
    )

    bundle = _compile(hits, max_items=3)

    # The two lowest-ranked hits are the only members of their sections, so they take a slot
    # before the second-ranked fact does.
    assert {hit.id for hit in bundle.hits} == {"fact-0", "trait-0", "cue-0"}
    assert bundle.omitted == 4


def test_selection_stays_in_rank_order_within_a_section() -> None:
    hits = tuple(
        _hit(f"fact-{index}", score=0.9 - index / 100, kind=MemoryKind.STATE) for index in range(4)
    )

    bundle = _compile(hits, max_items=3)

    assert [hit.id for hit in bundle.facts] == ["fact-0", "fact-1", "fact-2"]
    assert [hit.id for hit in bundle.hits] == ["fact-0", "fact-1", "fact-2"]
    assert bundle.omitted == 1


def test_item_and_character_budgets_are_never_exceeded() -> None:
    hits = tuple(
        _hit(f"fact-{index}", score=0.9 - index / 100, content="x" * 100, kind=MemoryKind.STATE)
        for index in range(10)
    )

    items = _compile(hits, max_items=4)
    assert len(items.hits) == 4
    assert items.omitted == 6

    chars = _compile(hits, max_chars=250)
    assert chars.chars == 200
    assert len(chars.hits) == 2
    assert chars.omitted == 8


def test_a_media_hit_is_charged_far_above_its_record_text() -> None:
    video = _hit(
        "clip",
        modality=Modality.VIDEO,
        assets=(
            AssetRef(
                id="asset-1",
                modality=Modality.VIDEO,
                media_type="video/mp4",
                size_bytes=1,
                sha256="a" * 64,
                path=Path("clip.mp4"),
            ),
        ),
    )
    text = _hit("note", content="a short note")

    assert evidence_cost(video) > evidence_cost(text)
    # The clip alone overruns a small budget, and the cheaper text still gets in behind it.
    bundle = _compile((video, text), max_chars=100)
    assert [hit.id for hit in bundle.hits] == ["note"]
    assert bundle.omitted == 1


# ---------------------------------------------------------------------------------------------
# Filters


def test_memory_types_and_min_confidence_filter_candidates() -> None:
    hits = (
        _hit("episode", memory_type=MemoryType.EPISODIC, kind=MemoryKind.EVENT),
        _hit("fact", memory_type=MemoryType.SEMANTIC, kind=MemoryKind.STATE),
    )

    typed = _compile(hits, memory_types=frozenset({MemoryType.EPISODIC}))
    assert [hit.id for hit in typed.hits] == ["episode"]
    assert typed.omitted == 0

    weak = _compile(
        (
            _hit("weak", kind=MemoryKind.STATE, confidence=0.4),
            _hit("strong", kind=MemoryKind.STATE, confidence=0.8),
            _hit("untyped"),
        ),
        min_confidence=0.5,
    )
    # A record with no typed context counts as fully confident rather than as an unknown.
    assert {hit.id for hit in weak.hits} == {"strong", "untyped"}


def test_freshness_anchors_on_event_end_then_event_start_then_creation() -> None:
    old = REFERENCE - timedelta(days=10)
    recent = REFERENCE - timedelta(hours=1)
    hits = (
        _hit("ended-recently", occurred_at=old, occurred_end=recent, created_at=old),
        _hit("started-recently", occurred_at=recent, created_at=old),
        _hit("written-recently", created_at=recent),
        _hit("stale", occurred_at=old, created_at=old),
    )

    bundle = _compile(hits, freshness=timedelta(days=1))

    assert {hit.id for hit in bundle.hits} == {
        "ended-recently",
        "started-recently",
        "written-recently",
    }
    assert bundle.omitted == 0


# ---------------------------------------------------------------------------------------------
# Conflicts, bounds, and frames


def test_disagreeing_states_in_one_lineage_produce_one_conflict() -> None:
    bundle = _compile(
        (
            _hit("first", score=0.9, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"),
            _hit("second", score=0.8, kind=MemoryKind.STATE, lineage_id="lineage-1", value="paris"),
            _hit("echo", score=0.7, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"),
            _hit("other", score=0.6, kind=MemoryKind.STATE, lineage_id="lineage-2", value="paris"),
            _hit("event", score=0.5, kind=MemoryKind.EVENT, lineage_id="lineage-3", value="paris"),
        )
    )

    assert len(bundle.conflicts) == 1
    conflict = bundle.conflicts[0]
    assert conflict.lineage_id == "lineage-1"
    assert (conflict.subject, conflict.predicate) == ("ana", "location")
    assert conflict.values == ("berlin", "paris")
    assert conflict.memory_ids == ("first", "second")


def test_bounds_and_frames_summarize_only_included_hits() -> None:
    start = REFERENCE - timedelta(hours=5)
    end = REFERENCE - timedelta(hours=4)
    bundle = _compile(
        (
            _hit(
                "kept",
                score=0.9,
                kind=MemoryKind.EVENT,
                occurred_at=start,
                occurred_end=end,
                frame_id="kitchen",
            ),
            _hit(
                "also",
                score=0.8,
                kind=MemoryKind.EVENT,
                occurred_at=REFERENCE - timedelta(hours=2),
                frame_id="hall",
            ),
            _hit("dropped", score=0.1, occurred_at=REFERENCE - timedelta(days=9)),
        ),
        max_items=2,
    )

    assert (bundle.occurred_from, bundle.occurred_until) == (
        start,
        REFERENCE - timedelta(hours=2),
    )
    assert bundle.frames == ("hall", "kitchen")
    assert bundle.omitted == 1


def test_an_empty_retrieval_compiles_an_empty_bundle() -> None:
    bundle = _compile(())

    assert bundle.hits == ()
    assert (bundle.occurred_from, bundle.occurred_until) == (None, None)
    assert (bundle.frames, bundle.conflicts, bundle.omitted, bundle.chars) == ((), (), 0, 0)
    assert "Omitted" not in bundle.render()


# ---------------------------------------------------------------------------------------------
# Rendering


def test_render_is_deterministic_and_names_every_included_id() -> None:
    bundle = _compile(
        (
            _hit("actor-1", score=0.9, kind=MemoryKind.ENTITY),
            _hit(
                "fact-1",
                score=0.8,
                kind=MemoryKind.STATE,
                lineage_id="lineage-1",
                value="berlin",
                valid_from=REFERENCE - timedelta(days=2),
                valid_until=REFERENCE,
            ),
            _hit("fact-2", score=0.7, kind=MemoryKind.STATE, lineage_id="lineage-1", value="paris"),
            _hit("episode-1", score=0.6, memory_type=MemoryType.EPISODIC, content="a\nb"),
            _hit("fact-3", score=0.5, kind=MemoryKind.STATE),
        ),
        max_items=4,
    )
    rendered = bundle.render()

    assert rendered == bundle.render()
    assert all(hit.id in rendered for hit in bundle.hits)
    assert "fact-3" not in rendered
    assert rendered.splitlines()[:4] == [
        "# Context: what is going on",
        "Each line is one memory: [id] content (confidence; validity).",
        f"Reference time: {REFERENCE.isoformat()}",
        f"Budget: {bundle.chars}/16000 chars, 4/4 items",
    ]
    assert "## Actors" in rendered and "## Facts" in rendered
    assert "- [actor-1] evidence actor-1 (confidence 0.90)" in rendered
    # A stored newline collapses so one memory stays one line.
    assert "- [episode-1] a b (confidence 1.00)" in rendered
    assert (
        "- [fact-1] evidence fact-1 (confidence 0.90; valid "
        f"{(REFERENCE - timedelta(days=2)).isoformat()} → {REFERENCE.isoformat()})"
    ) in rendered
    assert '- ana location: "berlin" [fact-1] vs "paris" [fact-2]' in rendered
    assert rendered.splitlines()[-1] == "Omitted: 1 lower-ranked candidates"


def test_render_keeps_the_section_order_fixed() -> None:
    bundle = _compile(
        (
            _hit("trait-1", kind=MemoryKind.TRAIT),
            _hit("cue-1", kind=MemoryKind.AFFECT),
            _hit("procedure-1", memory_type=MemoryType.PROCEDURAL),
            _hit("episode-1", memory_type=MemoryType.EPISODIC),
            _hit("fact-1", kind=MemoryKind.STATE),
            _hit("actor-1", kind=MemoryKind.ENTITY),
        )
    )

    headings = [line for line in bundle.render().splitlines() if line.startswith("## ")]
    assert headings == [
        "## Actors",
        "## Facts",
        "## Episodes",
        "## Procedures",
        "## Affect",
        "## Traits",
    ]


def test_an_open_validity_interval_renders_as_valid_from() -> None:
    bundle = _compile((_hit("from-only", score=0.9, kind=MemoryKind.STATE, valid_from=REFERENCE),))

    assert f"(confidence 0.90; valid from {REFERENCE.isoformat()})" in bundle.render()


# ---------------------------------------------------------------------------------------------
# Kernel integration


class _Former:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "compiler-test"
    formation_space = "compiler-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                FormationProposal(
                    kind=MemoryKind.TRAIT,
                    content="The user tends to prefer concise responses",
                    subject="user",
                    predicate="response_style",
                    value="concise",
                    confidence=0.7,
                ),
            )
            if "trait" in item.content.text
            else ()
            for item in inputs
        )

    def close(self) -> None:
        return None


def _memory(data_dir: Path) -> Memory:
    return Memory(
        data_dir,
        embedder=TinyEmbedder(),
        former=_Former(),
        minimum_relevance=0,
    )


def test_compile_reuses_the_retrieval_path_and_structures_what_it_returns(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        note = memory.add("the spare key is in the blue toolbox")
        walk = memory.add(
            "we walked to the harbour",
            memory_type=MemoryType.EPISODIC,
            occurred_at=REFERENCE - timedelta(hours=3),
        )

        bundle = memory.compile("what do you know", reference_at=REFERENCE)

        assert bundle.goal == "what do you know"
        assert bundle.reference_at == REFERENCE
        assert {hit.id for hit in bundle.hits} == {note.id, walk.id}
        assert [hit.id for hit in bundle.episodes] == [walk.id]
        assert [hit.id for hit in bundle.facts] == [note.id]
        assert bundle.occurred_from == REFERENCE - timedelta(hours=3)
        assert bundle.chars == len(note.content) + len(walk.content)
        assert note.id in bundle.render()


def test_compile_rejects_a_budget_that_is_not_one(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory, pytest.raises(ValidationError, match="ContextBudget"):
        memory.compile("anything", budget="6000")  # type: ignore[arg-type]


def test_a_forgotten_record_never_reaches_a_bundle(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        kept = memory.add("the red wrench is on the bench")
        gone = memory.add("the blue hammer is on the bench")
        # Forgetting has no public verb yet; the control plane owns it. This exercises the
        # read-path rule the compiler inherits from the shared retrieval kernel.
        assert memory._store.set_forgotten((gone.id,), forgotten_at=REFERENCE) == (gone.id,)

        bundle = memory.compile("what is on the bench", reference_at=REFERENCE)

        assert [hit.id for hit in bundle.hits] == [kept.id]
        assert memory.get(gone.id).forgotten_at == REFERENCE


def test_a_hidden_inferred_trait_never_reaches_a_bundle(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        # One model inference is not enough support, so the derived trait stays invisible.
        memory.add("trait evidence one")
        hidden = memory.compile("concise responses", reference_at=REFERENCE)
        assert hidden.traits == ()

        memory.add(
            "trait: I prefer concise responses",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        visible = memory.compile("concise responses", reference_at=REFERENCE)

    assert len(visible.traits) == 1
    assert visible.traits[0].context is not None
    assert visible.traits[0].context.kind is MemoryKind.TRAIT


def test_capabilities_reflect_the_injected_backends(tmp_path: Path) -> None:
    with _memory(tmp_path / "lean") as memory:
        lean = memory.capabilities
    assert lean.embedding == ATOMIC_MODALITIES
    assert (lean.generation_model, lean.transcription_space, lean.face_model) == (None, None, None)
    assert lean.formation_model is not None
    assert lean.consolidation_model is None

    with Memory(
        tmp_path / "full",
        embedder=TinyEmbedder(),
        answerer=_Answerer(),
        consolidator=ScriptedConsolidator(),
        decay_half_life_days=30.0,
    ) as memory:
        full = memory.capabilities
    assert full.generation_model == "compiler-test-answerer"
    assert full.formation_model is None
    assert full.consolidation_model == "consolidator-test"


class _Answerer:
    generation_capabilities = frozenset({Modality.TEXT})
    generation_model = "compiler-test-answerer"

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
        raise AssertionError("compile never generates text")

    def close(self) -> None:
        return None


def test_async_compile_and_capabilities_mirror_the_sync_surface(tmp_path: Path) -> None:
    async def run() -> tuple[ContextBundle, MemoryCapabilities]:
        async with AsyncMemory(
            tmp_path,
            embedder=TinyEmbedder(),
            former=_Former(),
            consolidator=ScriptedConsolidator(),
            minimum_relevance=0,
        ) as memory:
            await memory.add("the spare key is in the blue toolbox")
            return await memory.compile(
                "what do you know",
                budget=ContextBudget(max_items=1),
                reference_at=REFERENCE,
            ), memory.capabilities

    bundle, capabilities = asyncio.run(run())

    assert len(bundle.hits) == 1
    assert bundle.reference_at == REFERENCE
    assert capabilities.consolidation_model == "consolidator-test"
    assert capabilities.embedding == ATOMIC_MODALITIES


def test_the_cli_command_serializes_the_bundle(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add("the spare key is in the blue toolbox")
        arguments = _parser().parse_args(
            [
                "compile",
                "what do you know",
                "--max-items",
                "1",
                "--memory-type",
                "semantic",
                "--freshness-seconds",
                "86400",
                "--reference-at",
                REFERENCE.isoformat(),
            ]
        )
        document = _LOCAL["compile"](memory, arguments)

    assert json.loads(json.dumps(document))["budget"] == {
        "max_chars": 16_000,
        "max_items": 1,
        "memory_types": ["semantic"],
        "min_confidence": 0.0,
        "freshness_seconds": 86400.0,
    }
    assert len(document["facts"]) == 1  # type: ignore[arg-type]
    assert "Budget: " in str(document["rendered"])


# ---------------------------------------------------------------------------------------------
# Provisional actors


def test_a_provisional_actor_joins_the_actors_without_taking_a_hit_slot() -> None:
    """An unnamed person is reported beside the ranked actors, and costs nothing."""
    bundle = compile_context(
        "who is in the room",
        (
            _hit("actor", kind=MemoryKind.ENTITY),
            _hit("episode", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC),
        ),
        budget=ContextBudget(),
        reference_at=REFERENCE,
        provisional={"episode": ("identity_2", "identity_1"), "omitted": ("identity_3",)},
    )

    assert [entry.id for entry in bundle.actors if isinstance(entry, SearchHit)] == ["actor"]
    assert [
        (entry.identity_id, entry.memory_ids)
        for entry in bundle.actors
        if isinstance(entry, ProvisionalActor)
    ] == [("identity_1", ("episode",)), ("identity_2", ("episode",))]
    # Not a hit, not a budgeted item, and not the reason anything was omitted.
    assert [hit.id for hit in bundle.hits] == ["actor", "episode"]
    assert (bundle.omitted, bundle.chars) == (0, sum(evidence_cost(hit) for hit in bundle.hits))


def test_a_provisional_actor_of_omitted_evidence_is_not_reported() -> None:
    """The bundle never claims a person the reader cannot see the evidence for."""
    bundle = compile_context(
        "who is in the room",
        (_hit("expensive", content="x" * 500),),
        budget=ContextBudget(max_chars=10),
        reference_at=REFERENCE,
        provisional={"expensive": ("identity_1",)},
    )

    assert bundle.actors == ()
    assert bundle.omitted == 1


def test_rendering_labels_a_provisional_actor_plainly() -> None:
    bundle = compile_context(
        "who is in the room",
        (_hit("clip", kind=MemoryKind.ENTITY),),
        budget=ContextBudget(),
        reference_at=REFERENCE,
        provisional={"clip": ("identity_1",)},
    )
    rendered = bundle.render().splitlines()

    assert rendered[-1] == (
        "- [identity_1] unnamed person present (provisional identity; seen in [clip])"
    )
    assert "1/24 items" in bundle.render()
    assert bundle.render() == bundle.render()
