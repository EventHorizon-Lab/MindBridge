"""Label-free metamorphic checks for conflict-safe context delivery.

These tests exercise the public ``Memory`` SDK with the real SQLite and Zvec paths.  The model
boundaries are deterministic test doubles so a retrieval or compiler change is the only causal
variable; no benchmark answer or expected label is supplied to ingestion or retrieval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from mindbridge import (
    ContextBudget,
    ContextBundle,
    ContextUnknownKind,
    EmbedTask,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    Modality,
    ModelInput,
    ObservationContext,
    RetrievalScope,
)


class _RankedEmbedder:
    """Put a marked counterclaim below more than one full retrieval window."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "metamorphic-rank-test"
    embedding_space = "metamorphic-rank-test:4:l2-v1"
    embedding_dimension = 4

    def __init__(self) -> None:
        self.calls: list[tuple[EmbedTask, tuple[str, ...]]] = []

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        texts = tuple(value.text for value in inputs)
        self.calls.append((task, texts))
        if task is EmbedTask.QUERY:
            return tuple((1.0, 0.0, 0.0, 0.0) for _text in texts)
        vectors = []
        for text in texts:
            if "hidden-counterclaim" in text:
                vector = (-1.0, 0.0, 0.0, 0.0)
            elif "noise" in text:
                vector = (0.8, 0.6, 0.0, 0.0)
            else:
                vector = (1.0, 0.0, 0.0, 0.0)
            magnitude = math.sqrt(sum(component * component for component in vector))
            vectors.append(tuple(component / magnitude for component in vector))
        return tuple(vectors)

    def close(self) -> None:
        pass


class _ClaimFormer:
    """Turn ``claim|subject|predicate|value|surface`` observations into typed claims."""

    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "metamorphic-claim-test"
    formation_space = "metamorphic-claim-test:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        results: list[tuple[FormationProposal, ...]] = []
        for item in inputs:
            parts = item.content.text.split("|", 4)
            if len(parts) != 5 or parts[0] != "claim":
                results.append(())
                continue
            _marker, subject, predicate, value, surface = parts
            results.append(
                (
                    FormationProposal(
                        kind=MemoryKind.TRAIT,
                        content=surface,
                        basis=EvidenceBasis.USER_STATEMENT,
                        subject=subject,
                        predicate=predicate,
                        value=value,
                        confidence=0.95,
                        valid_from=item.context.valid_from,
                        valid_until=item.context.valid_until,
                    ),
                )
            )
        return tuple(results)

    def close(self) -> None:
        pass


def _claim(subject: str, value: str, surface: str, *, predicate: str = "preferred_city") -> str:
    return f"claim|{subject}|{predicate}|{value}|{surface}"


def _values(bundle: ContextBundle) -> set[str]:
    hits = bundle.hits
    return {
        hit.context.value
        for hit in hits
        if hit.context is not None
        and hit.context.kind in {MemoryKind.STATE, MemoryKind.TRAIT}
        and hit.context.value is not None
    }


def _memory(path: Path) -> tuple[Memory, _RankedEmbedder, _ClaimFormer]:
    embedder = _RankedEmbedder()
    former = _ClaimFormer()
    return (
        Memory(path, embedder=embedder, former=former, minimum_relevance=0),
        embedder,
        former,
    )


def test_low_rank_counterclaim_survives_window_and_budget_perturbations(tmp_path: Path) -> None:
    """A visible competitor cannot disappear merely because irrelevant records outrank it."""
    memory, embedder, former = _memory(tmp_path)
    with memory:
        sources = memory.add_many(
            (
                _claim("Ana", "Berlin", "anchor-target"),
                _claim("Ana", "Paris", "hidden-counterclaim"),
                *(f"noise|{index}" for index in range(110)),
            )
        )
        calls_after_ingest = len(embedder.calls)
        formation_calls_after_ingest = former.calls

        roomy = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000, max_items=8),
        )
        tight = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000, max_items=4),
        )

    assert len(roomy.conflicts) == len(tight.conflicts) == 1
    assert (
        set(roomy.conflicts[0].values)
        == set(tight.conflicts[0].values)
        == {
            "Berlin",
            "Paris",
        }
    )
    assert set(roomy.conflicts[0].memory_ids) == set(tight.conflicts[0].memory_ids)
    assert sources[1].id in {
        evidence_id
        for hit in roomy.hits
        if hit.context is not None
        for evidence_id in hit.context.evidence_ids
    }
    assert former.calls == formation_calls_after_ingest
    assert embedder.calls[calls_after_ingest:] == [
        (EmbedTask.QUERY, ("ranked preference",)),
        (EmbedTask.QUERY, ("ranked preference",)),
    ]


def test_duplicate_value_does_not_invent_an_additional_conflict_side(tmp_path: Path) -> None:
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        memory.add_many(
            (
                _claim("Ana", "Berlin", "anchor-target"),
                _claim("Ana", "Paris", "hidden-counterclaim"),
                _claim("Ana", "Berlin", "anchor-target repeated"),
            )
        )

        bundle = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000, max_items=8),
        )

    assert len(bundle.conflicts) == 1
    assert set(bundle.conflicts[0].values) == {"Berlin", "Paris"}
    assert len(bundle.conflicts[0].memory_ids) == 2


def test_deleting_counterevidence_removes_only_that_conflict_side(tmp_path: Path) -> None:
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        berlin, paris = memory.add_many(
            (
                _claim("Ana", "Berlin", "anchor-target"),
                _claim("Ana", "Paris", "hidden-counterclaim"),
            )
        )
        before = memory.compile("ranked preference", budget=ContextBudget(max_chars=40_000))
        assert len(before.conflicts) == 1

        conflict = before.conflicts[0]
        paris_claim = conflict.memory_ids[conflict.values.index("Paris")]
        assert memory.delete(paris_claim) is True
        after = memory.compile("ranked preference", budget=ContextBudget(max_chars=40_000))
        assert memory.get(paris.id) == paris

    assert after.conflicts == ()
    assert "Berlin" in _values(after)
    assert "Paris" not in _values(after)
    assert paris.id not in {
        evidence_id
        for hit in after.hits
        if hit.context is not None
        for evidence_id in hit.context.evidence_ids
    }
    assert berlin.id in {
        evidence_id
        for hit in after.hits
        if hit.context is not None
        for evidence_id in hit.context.evidence_ids
    }


def test_subject_swap_does_not_create_a_cross_identity_conflict(tmp_path: Path) -> None:
    """Changing the subject is an identity metamorphism, so the lineage must also change."""
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        memory.add_many(
            (
                _claim("Ana", "Berlin", "anchor-target"),
                _claim("Bo", "Paris", "anchor-target for Bo"),
            )
        )
        bundle = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000, max_items=8),
        )

    claims = [
        hit.context
        for hit in bundle.hits
        if hit.context is not None and hit.context.kind is MemoryKind.TRAIT
    ]
    assert {claim.subject for claim in claims} == {"Ana", "Bo"}
    assert len({claim.lineage_id for claim in claims}) == 2
    assert bundle.conflicts == ()


def test_valid_and_transaction_time_switch_the_visible_claim(tmp_path: Path) -> None:
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        memory.add(
            _claim("Ana", "Berlin", "anchor-target"),
            context=ObservationContext(
                basis=EvidenceBasis.USER_STATEMENT,
                valid_from=january,
            ),
        )
        known_before_correction = datetime.now(timezone.utc)
        memory.add(
            _claim("Ana", "Paris", "hidden-counterclaim"),
            context=ObservationContext(
                basis=EvidenceBasis.USER_STATEMENT,
                valid_from=february,
            ),
        )
        known_after_correction = datetime.now(timezone.utc)

        historical = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000),
            scope=RetrievalScope(valid_at=january, known_at=known_after_correction),
        )
        current = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000),
            scope=RetrievalScope(valid_at=march, known_at=known_after_correction),
        )
        before_correction = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=40_000),
            scope=RetrievalScope(valid_at=march, known_at=known_before_correction),
        )

    assert _values(historical) == {"Berlin"}
    assert _values(current) == {"Paris"}
    assert _values(before_correction) == {"Berlin"}
    assert historical.conflicts == current.conflicts == before_correction.conflicts == ()


def test_lineage_completion_cap_withholds_an_affirmative_claim(tmp_path: Path) -> None:
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        memory.add_many(
            tuple(_claim("Ana", f"city-{index:02d}", "anchor-target") for index in range(12))
        )
        bundle = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=80_000, max_items=40),
        )

    assert not any(
        hit.context is not None and hit.context.kind is MemoryKind.TRAIT for hit in bundle.hits
    )
    assert any(
        unknown.kind is ContextUnknownKind.EVIDENCE_UNAVAILABLE for unknown in bundle.unknowns
    )


def test_derived_anchor_cannot_reintroduce_an_unsafe_supporting_claim(tmp_path: Path) -> None:
    """Withholding a lineage also withholds closures that would render it as support."""
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        memory.add_many(
            tuple(
                _claim(
                    "Ana",
                    f"city-{index:02d}",
                    "anchor-target" if index == 0 else "hidden-counterclaim",
                )
                for index in range(12)
            )
        )
        first_claim = next(
            item
            for item in memory.list(limit=100).items
            if item.context is not None
            and item.context.kind is MemoryKind.TRAIT
            and item.context.value == "city-00"
        )
        derived_id = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(first_claim.id,),
                proposal=FormationProposal(
                    kind=MemoryKind.EVENT,
                    content="anchor-target derived conclusion",
                    confidence=0.8,
                ),
            )
        ).created_ids[0]
        memory.add_many(tuple(f"noise|closure|{index}" for index in range(110)))

        bundle = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=80_000, max_items=40),
        )

    assert derived_id not in {hit.id for hit in bundle.hits}
    assert not any(
        hit.context is not None and hit.context.kind is MemoryKind.TRAIT for hit in bundle.hits
    )
    assert any(
        unknown.kind is ContextUnknownKind.EVIDENCE_UNAVAILABLE for unknown in bundle.unknowns
    )


def test_many_same_value_repetitions_remain_answerable(tmp_path: Path) -> None:
    """A safety cap counts competing values, not harmless repetitions or historical volume."""
    memory, _embedder, _former = _memory(tmp_path)
    with memory:
        memory.add_many(
            tuple(
                _claim("Ana", "Berlin", f"anchor-target repetition {index}") for index in range(20)
            )
        )
        bundle = memory.compile(
            "ranked preference",
            budget=ContextBudget(max_chars=100_000, max_items=50),
        )

    assert _values(bundle) == {"Berlin"}
    assert bundle.conflicts == ()
    assert not any(
        unknown.kind is ContextUnknownKind.EVIDENCE_UNAVAILABLE for unknown in bundle.unknowns
    )
