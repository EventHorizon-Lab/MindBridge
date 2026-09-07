"""Metric arithmetic of the control-plane behaviour benchmark, on a known outcome.

The backends here are scripted, so what the loop will do is decided in this file. That is the
point: the assertions below are about the scoring, and a benchmark whose arithmetic is only ever
exercised against a real model cannot say whether a 0.5 came from the loop or from the metric.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from mindbridge import (
    EmbedTask,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    MemoryOutcome,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ModelInput,
)
from mindbridge.benchmarks import control_plane


class _TinyEmbedder:
    """Deterministic four-dimensional embedder; the scenario never depends on its ordering."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "tiny-control-plane"
    embedding_space = "tiny-control-plane:4:l2-v1"
    embedding_dimension = 4

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        vectors = []
        for value in inputs:
            digest = hashlib.sha256(value.text.encode()).digest()
            vector = tuple(1.0 + digest[index] / 255.0 for index in range(4))
            norm = math.sqrt(sum(component * component for component in vector))
            vectors.append(tuple(component / norm for component in vector))
        return tuple(vectors)

    def close(self) -> None:
        pass


class _ScriptedConsolidator:
    """Merges duplicate pairs, resolves contradictions, and optionally forgets a gold record."""

    consolidation_model = "scripted-control-plane"
    consolidation_recipe = "scripted-control-plane:v1"

    def __init__(
        self,
        *,
        forget_gold: bool = False,
        merge: bool = True,
        resolve: bool = True,
        wrong_side: bool = False,
    ) -> None:
        self.forget_gold = forget_gold
        self.merge = merge
        self.resolve = resolve
        self.wrong_side = wrong_side
        self.triggers: list[MemoryTrigger] = []

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.triggers.append(trigger)
        proposals: list[MemoryOperation] = []
        if self.resolve and trigger is MemoryTrigger.CONTRADICTION:
            proposals.extend(
                MemoryOperation(
                    intent=MemoryIntent.CORRECT,
                    target_ids=(stale,),
                    rationale="the newer claim is the one in force",
                )
                for stale in _disagreeing_claims(evidence, newest=self.wrong_side)
            )
        if self.merge:
            for pair in _drink_pairs(evidence):
                proposals.append(
                    MemoryOperation(
                        intent=MemoryIntent.CONSOLIDATE,
                        evidence_ids=pair,
                        target_ids=pair,
                        proposal=FormationProposal(
                            kind=MemoryKind.TRAIT,
                            content=f"{_person(evidence, pair[0])} drinks the same thing daily",
                            subject=_person(evidence, pair[0]),
                            predicate="routine",
                            value="same morning drink",
                            confidence=0.7,
                        ),
                    )
                )
        if self.forget_gold:
            gold = [record for record in evidence if " works as a " in record.content]
            if gold:
                proposals.append(
                    MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(gold[0].id,))
                )
        return tuple(proposals)

    def close(self) -> None:
        pass


def _person(evidence: Sequence[MemoryRecord], memory_id: str) -> str:
    for record in evidence:
        if record.id == memory_id:
            return record.content.split(" ", 1)[0]
    raise AssertionError(memory_id)


def _disagreeing_claims(evidence: Sequence[MemoryRecord], *, newest: bool) -> list[str]:
    """One side of each pair of disagreeing preference claims the trigger handed over."""
    by_subject: dict[str, list[MemoryRecord]] = {}
    for record in evidence:
        context = record.context
        if context is None or context.kind is not MemoryKind.RELATION:
            continue
        by_subject.setdefault(f"{context.subject}/{context.predicate}", []).append(record)
    stale = []
    for records in by_subject.values():
        if len({record.context.value for record in records if record.context}) < 2:
            continue
        # "now prefers" is what the newer observation says, so it also sorts the two claims.
        ordered = sorted(records, key=lambda item: " now prefers " in item.content)
        stale.append(ordered[-1].id if newest else ordered[0].id)
    return stale


def _drink_pairs(evidence: Sequence[MemoryRecord]) -> list[tuple[str, ...]]:
    """Group the two observations of one person's morning drink, newest id order fixed."""
    by_person: dict[str, list[MemoryRecord]] = {}
    for record in evidence:
        if " morning" not in record.content:
            continue
        by_person.setdefault(record.content.split(" ", 1)[0], []).append(record)
    return [
        tuple(sorted(record.id for record in records))
        for records in by_person.values()
        if len(records) == 2
    ]


def test_the_scenario_is_deterministic_and_declares_its_ground_truth(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_TinyEmbedder(), minimum_relevance=0) as memory:
        scenario = control_plane.build_scenario(memory, persons=3, seed=7)

    assert len(scenario.ingested) == 21
    assert len(scenario.duplicates) == 3
    assert len(scenario.contradictions) == 3
    assert len(scenario.gold) == 3
    assert scenario.failing_queries == tuple(
        f"what did {name} say about the archive keys" for name in ("Ana", "Bruno", "Chen")
    )
    assert not scenario.gold & {value for group in scenario.duplicates for value in group}


def test_a_loop_that_merges_exactly_the_injected_duplicates_scores_one(tmp_path: Path) -> None:
    consolidator = _ScriptedConsolidator()

    result = control_plane.run_benchmark(
        tmp_path,
        embedder=_TinyEmbedder(),
        consolidator=consolidator,
        persons=3,
        seed=7,
    )

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert MemoryTrigger.QUERY_FAILURE in consolidator.triggers
    assert MemoryTrigger.CONTRADICTION in consolidator.triggers
    assert metrics["consolidate_operations"] == 3
    assert metrics["consolidation_precision"] == pytest.approx(1.0)
    # Three merges and three CORRECTs, none refused: the claims the scenario derives are what
    # makes the CORRECT eligible at all, where a CORRECT on a raw observation is `not_derived`.
    assert metrics["applied_operations"] == 6
    deliberation = result["deliberation"]
    assert isinstance(deliberation, dict)
    assert deliberation["rejected"] == 0
    assert metrics["contradiction_recovery"] == pytest.approx(1.0)
    assert metrics["false_retirement"] == pytest.approx(0.0)
    assert metrics["rollback_success"] == pytest.approx(1.0)
    assert metrics["state_restored"] is True
    assert result["outcomes"] == {"confirmed": 6, "refuted": 0}


def test_forgetting_a_gold_record_is_scored_as_a_false_retirement(tmp_path: Path) -> None:
    """Nine retired records, three of them gold: the arithmetic that must not be rounded away."""
    consolidator = _ScriptedConsolidator(forget_gold=True)

    result = control_plane.run_benchmark(
        tmp_path,
        embedder=_TinyEmbedder(),
        consolidator=consolidator,
        persons=3,
        seed=7,
    )

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["consolidation_precision"] == pytest.approx(1.0)
    assert metrics["retired_records"] == 9
    assert metrics["false_retirement"] == pytest.approx(1 / 3)
    assert result["outcomes"] == {"confirmed": 6, "refuted": 3}


def test_correcting_the_claim_in_force_is_refuted_and_recovers_nothing(tmp_path: Path) -> None:
    """The wrong half of the flip: the log must not call it confirmed."""
    result = control_plane.run_benchmark(
        tmp_path,
        embedder=_TinyEmbedder(),
        consolidator=_ScriptedConsolidator(merge=False, wrong_side=True),
        persons=2,
        seed=7,
    )

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["applied_operations"] == 2
    assert metrics["contradiction_recovery"] == pytest.approx(0.0)
    assert result["outcomes"] == {"confirmed": 0, "refuted": 2}


def test_a_loop_that_proposes_nothing_leaves_the_rates_undefined(tmp_path: Path) -> None:
    """An undefined rate is not a perfect one, and a benchmark that reports 1.0 here lies."""
    result = control_plane.run_benchmark(
        tmp_path,
        embedder=_TinyEmbedder(),
        consolidator=_ScriptedConsolidator(merge=False, resolve=False),
        persons=2,
        seed=7,
    )

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["applied_operations"] == 0
    assert metrics["consolidation_precision"] is None
    assert metrics["false_retirement"] is None
    assert metrics["rollback_success"] is None
    assert metrics["contradiction_recovery"] == pytest.approx(0.0)
    assert result["outcomes"] == {"confirmed": 0, "refuted": 0}


def test_the_outcome_verdict_is_recorded_on_every_applied_operation(tmp_path: Path) -> None:
    """P5: `record_outcome` had no reader; the benchmark is one, so the log carries a judgement."""
    control_plane.run_benchmark(
        tmp_path,
        embedder=_TinyEmbedder(),
        consolidator=_ScriptedConsolidator(),
        persons=2,
        seed=7,
    )

    with Memory(tmp_path, embedder=_TinyEmbedder(), minimum_relevance=0) as memory:
        logged = memory.operations()

    # Only the loop's own rows are judged: the host's `apply()` rows that built the scenario's
    # claims are ground truth, not something the loop proposed for the ground truth to score.
    assert [record.trigger for record in logged].count(MemoryTrigger.MANUAL) == 4
    assert all(
        (record.outcome is MemoryOutcome.CONFIRMED) is (record.trigger is not MemoryTrigger.MANUAL)
        for record in logged
    )
