"""Focused checks for the optional compact Python context presentation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import pytest

from mindbridge import (
    AffectCue,
    ContextBudget,
    ContextBundle,
    ContextCitation,
    ContextConflict,
    ContextSymbolCoverage,
    ContextSymbolNamespace,
    ContextSymbolRole,
    ContextUnknown,
    ContextUnknownKind,
    EvidenceBasis,
    MemoryContext,
    MemoryKind,
    MemoryType,
    Modality,
    NamedActor,
    ProvisionalActor,
    SearchHit,
    ValidationError,
)

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
RAW = "a" * 64
DERIVED = "b" * 64
OUTSIDE_EVIDENCE = "c" * 64
AFFECT = "d" * 64
OUTSIDE_EVENT = "e" * 64
LITERAL_ID = "f" * 64
NAMING = "1" * 64
OUTSIDE_CONFLICT = "2" * 64
IDENTITY = "identity-alex"
OTHER_IDENTITY = "identity-visitor"


def _context(
    kind: MemoryKind,
    *,
    evidence_ids: tuple[str, ...] = (),
    cue: bool = False,
) -> MemoryContext:
    return MemoryContext(
        kind=kind,
        basis=EvidenceBasis.MODEL_INFERENCE,
        confidence=0.8,
        valid_from=NOW,
        valid_until=None,
        recorded_at=NOW,
        evidence_ids=evidence_ids,
        cue_modality=Modality.AUDIO if cue else None,
        valence=0.4 if cue else None,
        arousal=0.6 if cue else None,
    )


def _hit(
    identifier: str,
    content: str,
    *,
    context: MemoryContext | None = None,
    score: float = 0.8,
) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=content,
        score=score,
        created_at=NOW,
        memory_type=MemoryType.SEMANTIC,
        modality=Modality.TEXT,
        context=context,
    )


def _bundle(*, raw_id: str = RAW) -> ContextBundle:
    raw = _hit(
        raw_id,
        f"literal user tokens [m1] and {LITERAL_ID} stay byte-for-byte",
        score=0.7,
    )
    derived = _hit(
        DERIVED,
        "Alex knows the visitor",
        context=_context(MemoryKind.RELATION, evidence_ids=(raw_id, OUTSIDE_EVIDENCE)),
    )
    affect = AffectCue(
        id=AFFECT,
        content="Alex sounds relieved",
        score=0.6,
        created_at=NOW,
        memory_type=MemoryType.EPISODIC,
        modality=Modality.TEXT,
        context=_context(
            MemoryKind.AFFECT,
            evidence_ids=(raw_id, OUTSIDE_EVIDENCE),
            cue=True,
        ),
        event_ids=(OUTSIDE_EVENT,),
    )
    return ContextBundle(
        goal="who is here",
        reference_at=NOW,
        budget=ContextBudget(max_chars=16_000, max_items=24),
        actors=(
            NamedActor(
                identity_id=IDENTITY,
                name="Alex",
                memory_ids=(raw_id,),
                naming_assertion_id=NAMING,
            ),
            ProvisionalActor(identity_id=OTHER_IDENTITY, memory_ids=(DERIVED,)),
        ),
        relationships=(derived,),
        scene=(raw,),
        episodes=(),
        facts=(),
        procedures=(),
        affect=(affect,),
        traits=(),
        conflicts=(
            ContextConflict(
                lineage_id="alex-location",
                subject="alex",
                predicate="location",
                values=("inside", "outside"),
                memory_ids=(DERIVED, OUTSIDE_CONFLICT),
            ),
        ),
        unknowns=(
            ContextUnknown(
                kind=ContextUnknownKind.BUDGET_EXCLUDED,
                detail="diagnostic prose is not rewritten",
            ),
        ),
        occurred_from=None,
        occurred_until=None,
        frames=(),
        places=(),
        omitted=1,
        chars=12_000,
        elapsed_ms=2,
        deadline_exceeded=False,
    )


def test_compact_context_aliases_only_structural_ids_and_preserves_public_bundle() -> None:
    bundle = _bundle()
    rendered_before = bundle.render()
    hits_before = bundle.hits
    document_before = bundle.document()

    compact = bundle.compact()

    assert compact == bundle.compact()
    assert bundle.render() == rendered_before
    assert bundle.hits == hits_before
    assert bundle.document() == document_before
    assert "compact" not in bundle.document()
    assert compact.chars == len(compact.text)
    assert bundle.chars == 12_000
    assert len(bundle.hits) == 3
    assert "literal user tokens [m1]" in compact.text
    assert compact.text.count("[m1]") > 1
    assert LITERAL_ID in compact.text
    for structural_id in (
        RAW,
        DERIVED,
        OUTSIDE_EVIDENCE,
        AFFECT,
        OUTSIDE_EVENT,
        NAMING,
        OUTSIDE_CONFLICT,
        IDENTITY,
        OTHER_IDENTITY,
    ):
        assert structural_id not in compact.text
    assert "[i1] Alex present" in compact.text
    assert "named by [m2]" in compact.text
    assert "evidence [m1], [m4]" in compact.text
    assert "from [m1], [m4]" in compact.text
    assert "co-occurring events [m6]" in compact.text
    assert '"outside" [m7, not included]' in compact.text


def test_symbol_table_records_namespaces_roles_delivery_and_exact_round_trip() -> None:
    compact = _bundle().compact()
    symbols = {(item.namespace, item.stable_id): item for item in compact.symbols}

    assert symbols[(ContextSymbolNamespace.IDENTITY, IDENTITY)].symbol == "i1"
    assert symbols[(ContextSymbolNamespace.IDENTITY, OTHER_IDENTITY)].symbol == "i2"
    assert symbols[(ContextSymbolNamespace.MEMORY, RAW)].symbol == "m1"
    assert symbols[(ContextSymbolNamespace.MEMORY, NAMING)].symbol == "m2"
    assert symbols[(ContextSymbolNamespace.MEMORY, DERIVED)].symbol == "m3"
    assert symbols[(ContextSymbolNamespace.MEMORY, OUTSIDE_EVIDENCE)].symbol == "m4"
    assert symbols[(ContextSymbolNamespace.MEMORY, AFFECT)].symbol == "m5"
    assert symbols[(ContextSymbolNamespace.MEMORY, OUTSIDE_EVENT)].symbol == "m6"
    assert symbols[(ContextSymbolNamespace.MEMORY, OUTSIDE_CONFLICT)].symbol == "m7"
    assert symbols[(ContextSymbolNamespace.MEMORY, RAW)].roles == (
        ContextSymbolRole.OBSERVED_IN,
        ContextSymbolRole.EVIDENCE,
        ContextSymbolRole.HIT,
    )
    assert symbols[(ContextSymbolNamespace.MEMORY, DERIVED)].roles == (
        ContextSymbolRole.OBSERVED_IN,
        ContextSymbolRole.HIT,
        ContextSymbolRole.CONFLICT,
    )
    assert (
        symbols[(ContextSymbolNamespace.MEMORY, OUTSIDE_EVIDENCE)].coverage
        is ContextSymbolCoverage.REFERENCE_ONLY
    )
    assert symbols[(ContextSymbolNamespace.MEMORY, OUTSIDE_EVENT)].roles == (
        ContextSymbolRole.AFFECT_EVENT,
    )
    assert symbols[(ContextSymbolNamespace.MEMORY, NAMING)].roles == (
        ContextSymbolRole.NAMING_ASSERTION,
    )
    assert symbols[(ContextSymbolNamespace.MEMORY, RAW)].coverage is ContextSymbolCoverage.FULL
    assert (
        symbols[(ContextSymbolNamespace.MEMORY, OUTSIDE_EVIDENCE)].coverage
        is ContextSymbolCoverage.REFERENCE_ONLY
    )
    assert (
        symbols[(ContextSymbolNamespace.IDENTITY, IDENTITY)].coverage
        is ContextSymbolCoverage.REFERENCE_ONLY
    )
    assert all(
        compact.resolve(item.symbol, namespace=item.namespace) == item.stable_id
        for item in compact.symbols
    )
    assert compact.resolve_citation("m1") == ContextCitation(
        memory_id=RAW,
        coverage=ContextSymbolCoverage.FULL,
    )
    with pytest.raises(ValidationError, match="reference-only"):
        compact.resolve_citation("m4")
    with pytest.raises(ValidationError, match="identity symbols"):
        compact.resolve_citation("i1")


def test_symbols_are_bound_to_one_presentation_without_a_global_registry() -> None:
    first = _bundle().compact()
    second = _bundle(raw_id="9" * 64).compact()

    assert first.resolve("m1", namespace=ContextSymbolNamespace.MEMORY) == RAW
    assert second.resolve("m1", namespace=ContextSymbolNamespace.MEMORY) == "9" * 64
    with pytest.raises(ValidationError, match="unknown in this presentation"):
        first.resolve("m99", namespace=ContextSymbolNamespace.MEMORY)
    with pytest.raises(ValidationError, match="unknown in this presentation"):
        first.resolve("i1", namespace=ContextSymbolNamespace.MEMORY)
    with pytest.raises(FrozenInstanceError):
        first.chars = 0  # type: ignore[misc]

    with pytest.raises(ValidationError, match="coverage must be full or partial"):
        ContextCitation(memory_id=RAW, coverage=ContextSymbolCoverage.REFERENCE_ONLY)
    with pytest.raises(ValidationError, match="partial context citations require"):
        ContextCitation(memory_id=RAW, coverage=ContextSymbolCoverage.PARTIAL)


def test_id_addressed_goal_is_not_rewritten_or_claimed_semantically_equivalent() -> None:
    compact = replace(_bundle(), goal=f"explain memory {RAW}").compact()

    assert compact.text.splitlines()[0] == f"# Context: explain memory {RAW}"
    assert "- [m1] literal user tokens [m1]" in compact.text
    assert compact.resolve("m1", namespace=ContextSymbolNamespace.MEMORY) == RAW


def test_compact_presentation_never_repacks_the_bundle() -> None:
    bundle = _bundle()
    compact = bundle.compact()

    assert tuple(hit.id for hit in bundle.hits) == (DERIVED, RAW, AFFECT)
    assert compact.chars < len(bundle.render())
    assert bundle.budget == ContextBudget(max_chars=16_000, max_items=24)
