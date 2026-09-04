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
    NamedActor,
    ObservationContext,
    ProvisionalActor,
    RetrievalScope,
    SearchHit,
    SpatialAnchor,
    SpatialContext,
    ValidationError,
)
from mindbridge.cli import _LOCAL, _parser
from mindbridge.context import (
    _frame_cost,
    _heading_cost,
    bundle_cost,
    compile_context,
    evidence_cost,
    named_actor_cost,
    provisional_actor_cost,
)
from mindbridge.infrastructure.local.store import StoredAsset
from mindbridge.memory import _PreparedContent

REFERENCE = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
DAYS_10 = timedelta(days=10)


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
    place_id: str | None = None,
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
        place_id=place_id,
    )


def _frame_and_headings(bundle: ContextBundle, sections: Sequence[str]) -> int:
    return _frame_cost(bundle.goal, bundle.reference_at, bundle.budget) + sum(
        _heading_cost(name) for name in sections
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
        {"max_media_items": -1},
        {"max_media_items": True},
        {"max_media_items": 1.5},
        {"max_latency_ms": 0},
        {"max_latency_ms": 1.5},
        {"max_latency_ms": True},
    ),
)
def test_a_budget_that_cannot_bound_anything_is_rejected(invalid: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ContextBudget(**invalid)  # type: ignore[arg-type]


def test_budget_defaults_are_the_documented_ones() -> None:
    budget = ContextBudget()
    assert (budget.max_chars, budget.max_items) == (16000, 24)
    assert (budget.memory_types, budget.min_confidence, budget.freshness) == (None, 0.0, None)
    assert budget.max_latency_ms is None


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
            _hit("state", kind=MemoryKind.STATE, memory_type=MemoryType.SEMANTIC),
            _hit("relation", kind=MemoryKind.RELATION, memory_type=MemoryType.SEMANTIC),
            _hit("untyped", memory_type=MemoryType.SEMANTIC),
            _hit("episode", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC),
            _hit("procedure", kind=MemoryKind.RESPONSE_POLICY, memory_type=MemoryType.PROCEDURAL),
            _hit("cue", kind=MemoryKind.AFFECT, memory_type=MemoryType.EPISODIC),
            _hit("trait", kind=MemoryKind.TRAIT, memory_type=MemoryType.SEMANTIC),
        )
    )

    assert [hit.id for hit in bundle.actors if isinstance(hit, SearchHit)] == ["actor"]
    # Relationships and scene state are the doc's own words and now their own sections; `facts`
    # keeps only what is neither, so the two are no longer indistinguishable inside it.
    assert [hit.id for hit in bundle.relationships] == ["relation"]
    assert [hit.id for hit in bundle.scene] == ["state"]
    assert [hit.id for hit in bundle.facts] == ["untyped"]
    assert [hit.id for hit in bundle.episodes] == ["episode"]
    assert [hit.id for hit in bundle.procedures] == ["procedure"]
    assert [hit.id for hit in bundle.affect] == ["cue"]
    assert [hit.id for hit in bundle.traits] == ["trait"]
    assert len(bundle.hits) == 8


def _spanning_hits() -> tuple[SearchHit, ...]:
    return (
        *(
            _hit(f"fact-{index}", score=0.9 - index / 100, kind=MemoryKind.STATE)
            for index in range(5)
        ),
        _hit("trait-0", score=0.2, kind=MemoryKind.TRAIT),
        _hit("cue-0", score=0.1, kind=MemoryKind.AFFECT),
    )


def test_a_small_budget_stays_in_pure_rank_order() -> None:
    """Diversity never displaces rank when the budget cannot afford both.

    Three sections and three slots would spend two of them on the two lowest-ranked candidates,
    so the bundle would no longer be readable by score. Rank decides the whole of it instead.
    """
    bundle = _compile(_spanning_hits(), max_items=3)

    assert [hit.id for hit in bundle.hits] == ["fact-0", "fact-1", "fact-2"]
    assert (bundle.traits, bundle.affect) == ((), ())
    assert bundle.omitted == 4


def test_a_budget_that_affords_both_seats_every_section_it_spans() -> None:
    """Once the bottom half of `max_items` seats every spanned section, each one gets a slot."""
    bundle = _compile(_spanning_hits(), max_items=6)

    assert [hit.id for hit in bundle.hits] == [
        "fact-0",
        "fact-1",
        "fact-2",
        "fact-3",
        "trait-0",
        "cue-0",
    ]
    # The top half is rank alone; the floor round then buys the two sections it missed.
    assert [hit.id for hit in bundle.traits] == ["trait-0"]
    assert [hit.id for hit in bundle.affect] == ["cue-0"]
    assert bundle.omitted == 1


def test_selection_stays_in_rank_order_within_a_section() -> None:
    hits = tuple(
        _hit(f"fact-{index}", score=0.9 - index / 100, kind=MemoryKind.STATE) for index in range(4)
    )

    bundle = _compile(hits, max_items=3)

    assert [hit.id for hit in bundle.scene] == ["fact-0", "fact-1", "fact-2"]
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

    # `max_chars` prices the rendered evidence: the header, the section heading, and each
    # memory's whole `- [id] ... (confidence 1.00)` line, not `len(content)` alone.
    chars = _compile(hits, max_chars=500)
    assert len(chars.hits) == 2
    assert chars.omitted == 8
    assert chars.chars <= 500


def test_max_chars_bounds_the_rendered_evidence_and_not_only_the_record_text() -> None:
    """A bundle carrying no diagnostics renders inside `max_chars`, frame and headers included.

    The old accounting charged `len(content)` alone, so the header, the section headings and
    every `- [id] ... (confidence 1.00)` frame were spent outside the budget the caller declared
    and the docstring told callers to measure `render()` themselves.
    """
    hits = tuple(
        _hit(f"fact-{index}", score=0.9 - index / 100, content="x" * 60) for index in range(40)
    )

    for ceiling in (200, 400, 1000, 4000):
        bundle = _compile(hits, max_chars=ceiling)
        # `omitted` puts an unknown and a trailer below the evidence; measure the evidence.
        rendered = bundle.render().split("\n\n## Unknowns")[0]
        assert len(rendered) <= bundle.chars <= ceiling, ceiling

    # With nothing omitted there are no diagnostics at all, so the whole text is bounded.
    whole = _compile(hits[:2], max_chars=4000)
    assert whole.unknowns == ()
    assert len(whole.render()) <= whole.chars <= 4000


def test_max_media_items_bounds_grounded_parts_instead_of_their_price() -> None:
    """`0` compiles a text-only bundle; a positive bound caps the parts, not the characters."""
    clips = tuple(
        _hit(
            f"clip-{index}",
            score=0.9 - index / 100,
            modality=Modality.IMAGE,
            assets=(
                AssetRef(
                    id=f"asset-{index}",
                    modality=Modality.IMAGE,
                    media_type="image/png",
                    size_bytes=1,
                    sha256=f"{index}" * 64,
                    path=Path(f"frame-{index}.png"),
                ),
            ),
        )
        for index in range(4)
    )
    hits = (*clips, _hit("note", score=0.1, content="a short note"))

    # Unbounded: the default `max_chars` decides, and it buys some of the parts.
    assert [hit.id for hit in _compile(hits).hits] == [
        "clip-0",
        "clip-1",
        "clip-2",
        "clip-3",
        "note",
    ]
    # No media at all, whatever it costs and however it ranks.
    text_only = _compile(hits, max_media_items=0)
    assert [hit.id for hit in text_only.hits] == ["note"]
    assert text_only.omitted == 4
    # At most two image parts, and the text behind them still gets in.
    two = _compile(hits, max_media_items=2)
    assert [hit.id for hit in two.hits] == ["clip-0", "clip-1", "note"]
    assert two.omitted == 2


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
    bundle = _compile((video, text), max_chars=300)
    assert [hit.id for hit in bundle.hits] == ["note"]
    assert bundle.omitted == 1
    # ... but the default budget must be able to buy the most expensive single grounded part,
    # or no default compilation could ever reach a video memory.
    assert evidence_cost(video) < ContextBudget().max_chars
    assert [hit.id for hit in _compile((video, text)).hits] == ["clip", "note"]


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
    # A record with no typed context counts as fully confident rather than as an unknown: it is
    # an observation, which carries no inference to discount. It therefore survives the
    # strictest possible bound, and renders at the same 1.00 the filter used.
    assert {hit.id for hit in weak.hits} == {"strong", "untyped"}
    certain = _compile(
        (_hit("untyped"), _hit("inferred", kind=MemoryKind.STATE, confidence=0.99)),
        min_confidence=1.0,
    )
    assert [hit.id for hit in certain.hits] == ["untyped"]
    assert "- [untyped] evidence untyped (confidence 1.00)" in certain.render()


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


def test_a_conflict_survives_the_budget_dropping_one_side() -> None:
    bundle = _compile(
        (
            _hit(
                "berlin", score=0.9, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"
            ),
            _hit("paris", score=0.5, kind=MemoryKind.STATE, lineage_id="lineage-1", value="paris"),
        ),
        max_items=1,
    )

    assert [hit.id for hit in bundle.hits] == ["berlin"]
    assert bundle.conflicts[0].memory_ids == ("berlin", "paris")
    # The dropped side has no line of its own, so the conflict line says where it is not.
    assert '- ana location: "berlin" [berlin] vs "paris" [paris, not included]' in bundle.render()


def test_a_conflict_no_included_memory_asserts_is_not_this_bundles_disagreement() -> None:
    bundle = _compile(
        (
            _hit("kept", score=0.9),
            _hit(
                "berlin", score=0.5, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"
            ),
            _hit("paris", score=0.4, kind=MemoryKind.STATE, lineage_id="lineage-1", value="paris"),
        ),
        max_items=1,
    )

    assert [hit.id for hit in bundle.hits] == ["kept"]
    assert bundle.conflicts == ()


def test_a_filtered_out_side_of_a_conflict_stays_filtered_out() -> None:
    bundle = _compile(
        (
            _hit(
                "berlin", score=0.9, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"
            ),
            _hit(
                "paris",
                score=0.5,
                kind=MemoryKind.STATE,
                confidence=0.1,
                lineage_id="lineage-1",
                value="paris",
            ),
        ),
        min_confidence=0.5,
    )

    assert bundle.conflicts == ()


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


def test_the_spatial_summary_carries_symbolic_places_beside_metric_frames() -> None:
    bundle = _compile(
        (
            _hit("kept", score=0.9, kind=MemoryKind.EVENT, frame_id="map", place_id="kitchen"),
            _hit("also", score=0.8, kind=MemoryKind.EVENT, place_id="hall"),
            _hit("nowhere", score=0.7, kind=MemoryKind.EVENT),
        )
    )

    # `place_id` is the axis a household query uses, and every hit already carries it.
    assert bundle.places == ("hall", "kitchen")
    assert bundle.frames == ("map",)


def test_an_empty_retrieval_compiles_an_empty_bundle() -> None:
    bundle = _compile(())

    assert bundle.hits == ()
    assert (bundle.occurred_from, bundle.occurred_until) == (None, None)
    assert (bundle.frames, bundle.conflicts, bundle.omitted) == ((), (), 0)
    # An empty bundle still renders its header, so the header is what it charged.
    assert bundle.chars >= len(bundle.render())
    assert "Omitted" not in bundle.render()


# ---------------------------------------------------------------------------------------------
# Explicit unknowns and the latency deadline


def test_every_budget_bound_that_removed_evidence_is_named_with_its_count() -> None:
    bundle = _compile(
        (
            _hit("kept", score=0.9, kind=MemoryKind.STATE),
            _hit("wrong-type", score=0.8, memory_type=MemoryType.EPISODIC),
            _hit("weak", score=0.7, kind=MemoryKind.STATE, confidence=0.1),
            _hit("stale", score=0.6, kind=MemoryKind.STATE, created_at=REFERENCE - DAYS_10),
            _hit("crowded", score=0.5, kind=MemoryKind.STATE),
        ),
        max_items=1,
        memory_types=frozenset({MemoryType.SEMANTIC, MemoryType.PROCEDURAL}),
        min_confidence=0.5,
        freshness=timedelta(days=1),
    )

    assert [(item.kind.value, item.detail) for item in bundle.unknowns] == [
        ("budget_excluded", "1 candidates below the requested minimum confidence"),
        ("budget_excluded", "1 candidates did not fit 1 items and 16000 chars"),
        ("budget_excluded", "1 candidates older than the requested freshness window"),
        ("budget_excluded", "1 candidates outside the requested memory types"),
        (
            "section_empty",
            "the affect section is empty: memory_types kept only procedural, semantic, and"
            " affect carries only episodic memory",
        ),
        (
            "section_empty",
            "the episodes section is empty: memory_types kept only procedural, semantic, and"
            " episodes carries only episodic memory",
        ),
        ("section_empty", "the procedures section is empty: no procedural memory was included"),
    ]
    # Ordering is stable, so a caller can diff two bundles.
    assert (
        bundle.unknowns
        == _compile(
            (
                _hit("kept", score=0.9, kind=MemoryKind.STATE),
                _hit("wrong-type", score=0.8, memory_type=MemoryType.EPISODIC),
                _hit("weak", score=0.7, kind=MemoryKind.STATE, confidence=0.1),
                _hit("stale", score=0.6, kind=MemoryKind.STATE, created_at=REFERENCE - DAYS_10),
                _hit("crowded", score=0.5, kind=MemoryKind.STATE),
            ),
            max_items=1,
            memory_types=frozenset({MemoryType.SEMANTIC, MemoryType.PROCEDURAL}),
            min_confidence=0.5,
            freshness=timedelta(days=1),
        ).unknowns
    )


def test_a_memory_type_filter_names_the_sections_it_emptied() -> None:
    """A `{semantic}` request loses affect and procedures, and the bundle says which filter did.

    Kind `affect` is stored `episodic` and kind `response_policy` `procedural`, so those two
    sections cannot be filled under a `{semantic}` request however well their records rank. The
    reader has to be able to map the loss back to the section, not only to the type.
    """
    bundle = _compile(
        (
            _hit("fact", score=0.9),
            _hit("mood", score=0.8, kind=MemoryKind.AFFECT, memory_type=MemoryType.EPISODIC),
            _hit("recipe", score=0.7, memory_type=MemoryType.PROCEDURAL),
        ),
        memory_types=frozenset({MemoryType.SEMANTIC}),
    )

    assert bundle.facts == (bundle.hits[0],)
    assert (bundle.affect, bundle.procedures) == ((), ())
    assert [item.detail for item in bundle.unknowns if item.kind.value == "section_empty"] == [
        "the affect section is empty: memory_types kept only semantic, and affect carries only"
        " episodic memory",
        "the episodes section is empty: memory_types kept only semantic, and episodes carries"
        " only episodic memory",
        "the procedures section is empty: memory_types kept only semantic, and procedures"
        " carries only procedural memory",
    ]


def test_a_full_candidate_window_is_named_beside_what_the_bundle_lost() -> None:
    hits = tuple(_hit(f"hit-{index}", score=0.9 - index / 100) for index in range(4))

    def _window(limit: int | None, **budget: object) -> tuple[str, ...]:
        bundle = compile_context(
            "what is going on",
            hits,
            budget=ContextBudget(**budget),  # type: ignore[arg-type]
            reference_at=REFERENCE,
            candidate_limit=limit,
        )
        return tuple(item.detail for item in bundle.unknowns if item.kind.value == exhausted)

    exhausted = "candidates_exhausted"
    assert _window(4, max_items=2) == (
        "retrieval filled its 4 candidate window, so the counts above bound what this window"
        " held and not what the store holds",
    )
    # Room left in the window means the ranking really did end where the bundle says it did.
    assert _window(8, max_items=2) == ()
    # A full window that cost the bundle nothing is not an unknown.
    assert _window(4) == ()
    # A caller compiling hits it retrieved itself declares no window and is told about none.
    assert _window(None, max_items=2) == ()


def test_a_bundle_that_lost_nothing_reports_no_unknowns() -> None:
    bundle = _compile((_hit("kept", kind=MemoryKind.STATE),))

    assert bundle.unknowns == ()
    assert "## Unknowns" not in bundle.render()


def test_the_renderer_names_every_unknown() -> None:
    bundle = _compile((_hit("kept"), _hit("crowded", score=0.1)), max_items=1)

    rendered = bundle.render()

    assert "## Unknowns" in rendered
    assert "- budget_excluded: 1 candidates did not fit 1 items and 16000 chars" in rendered


def test_the_deadline_skips_optional_enrichment_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline is a checkpoint, never a cancellation: sections survive, enrichment does not."""
    hits = (
        _hit("first", score=0.9, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"),
        _hit("second", score=0.8, kind=MemoryKind.STATE, lineage_id="lineage-1", value="paris"),
    )
    # Two readings: the checkpoint before optional enrichment, then the reported elapsed time.
    clock = iter((0.5, 0.75))
    monkeypatch.setattr("mindbridge.context.perf_counter", lambda: next(clock))

    bundle = compile_context(
        "what is going on",
        hits,
        budget=ContextBudget(max_latency_ms=100),
        reference_at=REFERENCE,
        started_at=0.0,
    )

    assert [hit.id for hit in bundle.scene] == ["first", "second"]
    assert bundle.conflicts == ()
    assert bundle.deadline_exceeded is True
    assert bundle.elapsed_ms == 750
    assert [(item.kind.value, item.detail) for item in bundle.unknowns] == [
        ("stage_skipped", "conflict detection was skipped after the 100 ms deadline passed")
    ]


def test_a_deadline_that_holds_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = (
        _hit("first", score=0.9, kind=MemoryKind.STATE, lineage_id="lineage-1", value="berlin"),
        _hit("second", score=0.8, kind=MemoryKind.STATE, lineage_id="lineage-1", value="paris"),
    )
    clock = iter((0.01, 0.02))
    monkeypatch.setattr("mindbridge.context.perf_counter", lambda: next(clock))

    bundle = compile_context(
        "what is going on",
        hits,
        budget=ContextBudget(max_latency_ms=100),
        reference_at=REFERENCE,
        started_at=0.0,
    )

    assert len(bundle.conflicts) == 1
    assert (bundle.deadline_exceeded, bundle.elapsed_ms, bundle.unknowns) == (False, 20, ())


def test_a_bundle_without_a_deadline_still_reports_what_it_spent() -> None:
    bundle = _compile((_hit("kept"),))

    assert bundle.deadline_exceeded is False
    assert bundle.elapsed_ms >= 0


def test_a_scope_that_matched_nothing_is_reported_as_an_unknown(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add("the spare key is in the blue toolbox")

        bundle = memory.compile(
            "what do you know",
            reference_at=REFERENCE,
            scope=RetrievalScope(place_id="workshop"),
        )

    assert bundle.hits == ()
    assert [(item.kind.value, item.detail) for item in bundle.unknowns] == [
        ("scope_empty", "no memory matched the requested scope: place workshop")
    ]


def test_a_scope_that_matched_is_not_reported(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add(
            "the spare key is in the blue toolbox",
            context=ObservationContext(place_id="workshop"),
        )

        bundle = memory.compile(
            "what do you know",
            reference_at=REFERENCE,
            scope=RetrievalScope(place_id="workshop"),
        )

    assert bundle.places == ("workshop",)
    assert bundle.unknowns == ()


def test_goal_media_the_embedder_cannot_take_natively_is_reported(tmp_path: Path) -> None:
    """A text-only composition still answers a spoken goal, from a transcript rather than audio."""

    class TextOnly(TinyEmbedder):
        embedding_capabilities = frozenset({Modality.TEXT})

    digest = "a" * 64
    audio = StoredAsset(
        asset_id=digest,
        modality="audio",
        mime_type="audio/wav",
        size_bytes=1,
        sha256=digest,
        relative_path=f"assets/aa/{digest}",
        created_at=REFERENCE,
    )
    with Memory(tmp_path, embedder=TextOnly(), minimum_relevance=0) as memory:
        unknowns = memory._request_unknowns(
            _PreparedContent(
                text="what do you know",
                assets=(audio,),
                modality=Modality.OMNI,
                canonical_parts=(),
            ),
            None,
            (),
        )

    assert [(item.kind.value, item.detail) for item in unknowns] == [
        (
            "modality_unsupported",
            "the goal's audio was not embedded natively; embedding accepts text,"
            " so only derived text could have matched",
        )
    ]


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
    assert "## Actors" in rendered and "## Scene" in rendered
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
            _hit("fact-1"),
            _hit("state-1", kind=MemoryKind.STATE),
            _hit("relation-1", kind=MemoryKind.RELATION),
            _hit("actor-1", kind=MemoryKind.ENTITY),
        )
    )

    headings = [line for line in bundle.render().splitlines() if line.startswith("## ")]
    assert headings == [
        "## Actors",
        "## Relationships",
        "## Scene",
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
        assert len(bundle.render()) <= bundle.chars <= bundle.budget.max_chars
        assert note.id in bundle.render()


def test_a_type_only_budget_reaches_past_the_window_the_common_types_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`memory_types` is pushed into the index, so a rare type gets its own candidate depth.

    Filtering after retrieval let sixty episodes crowd the one procedural record out of the
    window, and the bundle came back empty with only a `candidates_exhausted` unknown to explain
    it. The narrowed rerank window here is what a real store reaches with more records.
    """
    monkeypatch.setattr("mindbridge.memory._RERANK_CANDIDATES", 4)
    with _memory(tmp_path) as memory:
        memory.add_many(
            [f"we walked to the harbour on day {index}" for index in range(60)],
            memory_type=MemoryType.EPISODIC,
        )
        recipe = memory.add(
            "to open the toolbox, turn the blue latch twice",
            memory_type=MemoryType.PROCEDURAL,
        )

        # The record is nowhere near the top of the unfiltered ranking for this goal.
        ranked = memory.search("we walked to the harbour", limit=4)
        assert recipe.id not in {hit.id for hit in ranked}

        bundle = memory.compile(
            "we walked to the harbour",
            budget=ContextBudget(max_items=1, memory_types=frozenset({MemoryType.PROCEDURAL})),
            reference_at=REFERENCE,
        )

        assert [hit.id for hit in bundle.procedures] == [recipe.id]


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
        "max_chars": 16000,
        "max_items": 1,
        "max_media_items": None,
        "memory_types": ["semantic"],
        "min_confidence": 0.0,
        "freshness_seconds": 86400.0,
        "max_latency_ms": None,
    }
    assert len(document["facts"]) == 1  # type: ignore[arg-type]
    assert "Budget: " in str(document["rendered"])


# ---------------------------------------------------------------------------------------------
# Provisional actors


def test_a_provisional_actor_joins_the_actors_without_taking_a_hit_slot() -> None:
    """An unnamed person is reported beside the ranked actors, priced like everything else."""
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
    actors = [
        (entry.identity_id, entry.memory_ids)
        for entry in bundle.actors
        if isinstance(entry, ProvisionalActor)
    ]
    assert actors == [("identity_1", ("episode",)), ("identity_2", ("episode",))]
    # Not a hit and not a budgeted item slot, but not the reason anything was omitted either.
    assert [hit.id for hit in bundle.hits] == ["actor", "episode"]
    assert bundle.omitted == 0
    provisional_actors = [entry for entry in bundle.actors if isinstance(entry, ProvisionalActor)]
    assert bundle.chars == (
        sum(bundle_cost(hit) for hit in bundle.hits)
        + sum(provisional_actor_cost(actor) for actor in provisional_actors)
        + _frame_and_headings(bundle, ("actors", "episodes"))
    )
    assert len(bundle.render()) <= bundle.chars <= bundle.budget.max_chars


def test_an_actor_line_is_dropped_rather_than_given_away_free_when_it_does_not_fit() -> None:
    """A tight budget drops an actor line the same way it drops an oversized hit."""
    hit = _hit("actor", kind=MemoryKind.ENTITY, content="short")
    # The exact price of the hit alone, with no room left over for an actor line, which always
    # costs at least one more character.
    unnamed = compile_context(
        "who is in the room", (hit,), budget=ContextBudget(), reference_at=REFERENCE
    )
    bundle = compile_context(
        "who is in the room",
        (hit,),
        budget=ContextBudget(max_chars=unnamed.chars),
        reference_at=REFERENCE,
        provisional={"actor": ("identity_1",)},
    )

    assert bundle.actors == (hit,)
    assert any(
        unknown.kind.value == "budget_excluded" and "actor line" in unknown.detail
        for unknown in bundle.unknowns
    )
    # The dropped actor never inflates `chars`; only the `## Unknowns` block that explains the
    # drop sits outside the ceiling, same as it does for a dropped hit.
    assert bundle.chars <= bundle.budget.max_chars


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


# ---------------------------------------------------------------------------------------------
# Named actors


def test_a_named_actor_surfaces_without_its_naming_assertion_in_top_k() -> None:
    """The identity edge names somebody even when the naming assertion itself lost its slot."""
    episode = _hit("episode", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC, score=0.9)
    naming = _hit("naming", kind=MemoryKind.ENTITY, score=0.1)
    bundle = compile_context(
        "who is in the room",
        (episode, naming),
        budget=ContextBudget(max_items=1),
        reference_at=REFERENCE,
        named={"episode": (("identity_1", "Li", "naming"),)},
    )

    assert [hit.id for hit in bundle.hits] == ["episode"]
    named_actors = [entry for entry in bundle.actors if isinstance(entry, NamedActor)]
    assert named_actors == [
        NamedActor(
            identity_id="identity_1",
            name="Li",
            memory_ids=("episode",),
            naming_assertion_id="naming",
        )
    ]
    assert not any(isinstance(entry, ProvisionalActor) for entry in bundle.actors)


def test_an_unnamed_identity_still_yields_a_provisional_actor_beside_a_named_one() -> None:
    """A named identity and an unrelated unnamed one coexist without interfering."""
    bundle = compile_context(
        "who is in the room",
        (_hit("episode", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC),),
        budget=ContextBudget(),
        reference_at=REFERENCE,
        named={"episode": (("identity_1", "Li", "naming"),)},
        provisional={"episode": ("identity_2",)},
    )

    assert [type(entry).__name__ for entry in bundle.actors] == ["NamedActor", "ProvisionalActor"]
    assert bundle.actors[0].identity_id == "identity_1"  # type: ignore[union-attr]
    assert bundle.actors[1].identity_id == "identity_2"  # type: ignore[union-attr]


def test_a_named_identity_is_never_also_reported_provisional() -> None:
    """A stale `provisional` entry for an identity `named` already reports is dropped."""
    bundle = compile_context(
        "who is in the room",
        (_hit("episode", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC),),
        budget=ContextBudget(),
        reference_at=REFERENCE,
        named={"episode": (("identity_1", "Li", "naming"),)},
        provisional={"episode": ("identity_1",)},
    )

    assert [type(entry).__name__ for entry in bundle.actors] == ["NamedActor"]


def test_a_named_actor_aggregates_every_memory_that_carries_its_edge() -> None:
    """The same identity observed by two included memories is one actor, not two."""
    bundle = compile_context(
        "who is in the room",
        (
            _hit("first", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC, score=0.9),
            _hit("second", kind=MemoryKind.EVENT, memory_type=MemoryType.EPISODIC, score=0.8),
        ),
        budget=ContextBudget(),
        reference_at=REFERENCE,
        named={
            "first": (("identity_1", "Li", "naming"),),
            "second": (("identity_1", "Li", "naming"),),
        },
    )

    named_actors = [entry for entry in bundle.actors if isinstance(entry, NamedActor)]
    assert named_actors == [
        NamedActor(
            identity_id="identity_1",
            name="Li",
            memory_ids=("first", "second"),
            naming_assertion_id="naming",
        )
    ]
    # One actor line however many memories carried its edge, priced once.
    assert bundle.chars == sum(bundle_cost(hit) for hit in bundle.hits) + named_actor_cost(
        named_actors[0]
    ) + _frame_and_headings(bundle, ("episodes",))


def test_a_named_actor_of_omitted_evidence_is_not_reported() -> None:
    """The bundle never claims a name for evidence the reader cannot see."""
    bundle = compile_context(
        "who is in the room",
        (_hit("expensive", content="x" * 500),),
        budget=ContextBudget(max_chars=10),
        reference_at=REFERENCE,
        named={"expensive": (("identity_1", "Li", "naming"),)},
    )

    assert bundle.actors == ()
    assert bundle.omitted == 1


def test_rendering_labels_a_named_actor_with_its_name_and_provenance() -> None:
    bundle = compile_context(
        "who is in the room",
        (_hit("clip", kind=MemoryKind.ENTITY),),
        budget=ContextBudget(),
        reference_at=REFERENCE,
        named={"clip": (("identity_1", "Li", "naming"),)},
    )
    rendered = bundle.render().splitlines()

    # The ranked entity hit renders first, the named actor beside it in the same section.
    assert rendered[-1] == "- [identity_1] Li present (seen in [clip]; named by [naming])"
    assert bundle.render() == bundle.render()
