"""Kernel contract for the agentic memory control plane.

Every test drives the public SDK. The backend here only proposes; what these assert is that the
kernel validates each proposal, applies exactly the effect its intent allows, logs it once, and
can reverse it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    AsyncMemory,
    ConsolidationBackend,
    ConsolidationReport,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryPlugins,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ModelError,
    ObservationContext,
    ValidationError,
)
from mindbridge.control import operation_key
from mindbridge.infrastructure.local import LocalStore
from mindbridge.infrastructure.local.store import StoredOperation

OCCURRED = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)


class ScriptedConsolidator:
    """Replays a fixed script of operations, resolving placeholders against shown evidence."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self, *scripts: Sequence[MemoryOperation]) -> None:
        self._scripts = [tuple(script) for script in scripts]
        self.calls: list[tuple[tuple[str, ...], MemoryTrigger]] = []

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.calls.append((tuple(record.id for record in evidence), trigger))
        if not self._scripts:
            return ()
        return self._scripts.pop(0)

    def close(self) -> None:
        pass


def _trait(subject: str, value: str, *, confidence: float = 0.6) -> FormationProposal:
    return FormationProposal(
        kind=MemoryKind.TRAIT,
        content=f"{subject} is {value}",
        subject=subject,
        predicate="disposition",
        value=value,
        confidence=confidence,
    )


def _memory(tmp_path: Path, consolidator: object | None = None, **kwargs: object) -> Memory:
    return Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=consolidator,  # type: ignore[arg-type]
        minimum_relevance=0,
        **kwargs,  # type: ignore[arg-type]
    )


def _observations(memory: Memory, *contents: str) -> tuple[MemoryRecord, ...]:
    return tuple(
        memory.add(
            content,
            occurred_at=OCCURRED + timedelta(minutes=index),
            context=ObservationContext(basis=EvidenceBasis.OBSERVATION, source_id=f"cam-{index}"),
        )
        for index, content in enumerate(contents)
    )


# ---------------------------------------------------------------------------------------------
# Vocabulary


def test_operation_shape_follows_its_intent() -> None:
    proposal = _trait("Ana", "patient")
    assert MemoryOperation(
        intent=MemoryIntent.CONSOLIDATE, evidence_ids=("a", "b"), proposal=proposal
    ).evidence_ids == ("a", "b")
    assert MemoryOperation(
        intent=MemoryIntent.REINFORCE, target_ids=("t",), evidence_ids=("a", "a")
    ).evidence_ids == ("a",)
    # Consolidation forgetting: a consolidation may name sources of its own to retire.
    assert MemoryOperation(
        intent=MemoryIntent.CONSOLIDATE,
        evidence_ids=("a", "b"),
        target_ids=("a",),
        proposal=proposal,
    ).target_ids == ("a",)

    malformed: tuple[dict[str, object], ...] = (
        {"intent": MemoryIntent.CONSOLIDATE, "evidence_ids": ("a",)},
        {"intent": MemoryIntent.CONSOLIDATE, "proposal": proposal},
        {"intent": MemoryIntent.CONSOLIDATE, "proposal": proposal, "target_ids": ("t",)},
        {"intent": MemoryIntent.REINFORCE, "evidence_ids": ("a",)},
        {"intent": MemoryIntent.REINFORCE, "target_ids": ("t", "u"), "evidence_ids": ("a",)},
        {"intent": MemoryIntent.REINFORCE, "target_ids": ("t",)},
        {"intent": MemoryIntent.REINFORCE, "target_ids": ("t",), "proposal": proposal},
        {"intent": MemoryIntent.CORRECT},
        {"intent": MemoryIntent.CORRECT, "target_ids": ("t",), "evidence_ids": ("a",)},
        {"intent": MemoryIntent.FORGET},
        {"intent": MemoryIntent.FORGET, "target_ids": (" ",)},
        {"intent": "shred", "target_ids": ("t",)},
    )
    for operation in malformed:
        with pytest.raises(ValidationError):
            MemoryOperation(**operation)  # type: ignore[arg-type]


def test_an_empty_report_is_valid_and_rejections_keep_their_reason() -> None:
    operation = MemoryOperation(intent=MemoryIntent.FORGET, target_ids=("t",))
    report = ConsolidationReport(rejected=((operation, "duplicate"),))

    assert report.operations == ()
    assert report.rejected == ((operation, "duplicate"),)
    with pytest.raises(ValidationError):
        ConsolidationReport(rejected=((operation, " "),))


# ---------------------------------------------------------------------------------------------
# Backend injection


def test_the_consolidator_is_a_protocol_slot_closed_once_with_the_others(tmp_path: Path) -> None:
    class Counter(ScriptedConsolidator):
        closed = 0

        def close(self) -> None:
            type(self).closed += 1

    consolidator = Counter()
    assert isinstance(consolidator, ConsolidationBackend)
    memory = Memory.from_plugins(
        tmp_path / "plugged",
        plugins=MemoryPlugins(embedder=TinyEmbedder(), consolidator=consolidator),
    )
    memory.close()
    memory.close()

    assert Counter.closed == 1
    with pytest.raises(ValidationError):
        MemoryPlugins(embedder=TinyEmbedder(), consolidator=object())  # type: ignore[arg-type]


def test_consolidate_without_a_backend_reports_the_missing_capability(tmp_path: Path) -> None:
    with _memory(tmp_path / "bare") as memory:
        with pytest.raises(ModelError) as failure:
            memory.consolidate()
        assert failure.value.reason == "backend_not_configured"


# ---------------------------------------------------------------------------------------------
# CONSOLIDATE


def test_two_independent_sources_consolidate_into_one_derived_record(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "derive", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again, calmly")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                    rationale="two independent waits",
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert report.rejected == ()
        assert len(report.operations) == 1
        record = report.operations[0]
        assert record.operation.intent is MemoryIntent.CONSOLIDATE
        assert record.model_id == "consolidator-test"
        assert record.recipe == "consolidator-test:v1"
        assert len(record.created_ids) == 1

        derived = memory.get(record.created_ids[0])
        assert derived.context is not None
        assert derived.context.kind is MemoryKind.TRAIT
        assert set(derived.context.evidence_ids) == {first.id, second.id}
        # Noisy-OR over two independent sources, not the single-source proposal confidence.
        assert derived.context.confidence == pytest.approx(1 - 0.4 * 0.4)
        # A model-inferred trait needs two independent sources to be visible at all.
        assert derived.context.visible is True
        assert any(hit.id == derived.id for hit in memory.search("patient", limit=10))


def test_re_proposing_the_same_consolidation_is_rejected_as_a_duplicate(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "dupe", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        operation = MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(first.id, second.id),
            proposal=_trait("Ana", "patient"),
        )
        consolidator._scripts.extend(((operation,), (operation,)))

        first_pass = memory.consolidate(evidence_ids=(first.id, second.id))
        second_pass = memory.consolidate(evidence_ids=(first.id, second.id))

        assert len(first_pass.operations) == 1
        assert second_pass.operations == ()
        assert [reason for _operation, reason in second_pass.rejected] == ["duplicate"]
        assert len(memory.operations()) == 1


def test_consolidation_cannot_cite_evidence_outside_the_shown_set(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "unshown", consolidator) as memory:
        shown, hidden = _observations(memory, "Ana waited", "unrelated noise")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(shown.id, hidden.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(shown.id,))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["evidence_not_shown"]
        assert memory.operations() == ()


def test_consolidation_requires_the_affect_cue_modality_to_be_present(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "affect", consolidator) as memory:
        (source,) = _observations(memory, "Ana said she was pleased")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(source.id,),
                    proposal=FormationProposal(
                        kind=MemoryKind.AFFECT,
                        content="Ana sounded pleased",
                        subject="Ana",
                        value="pleased",
                        confidence=0.7,
                        cue_modality=Modality.AUDIO,
                    ),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(source.id,))

        assert [reason for _operation, reason in report.rejected] == ["invalid_proposal"]


def test_rolling_back_a_consolidation_removes_the_derived_record(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "undo-derive", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        record = memory.consolidate(evidence_ids=(first.id, second.id)).operations[0]
        derived_id = record.created_ids[0]

        assert memory.rollback(record.operation_id) is True
        assert memory.rollback(record.operation_id) is False
        assert memory.operations()[0].rolled_back_at is not None
        with pytest.raises(MemoryNotFoundError):
            memory.get(derived_id)
        # The cited observations are evidence and survive their derived record.
        assert memory.get(first.id).id == first.id
        assert memory.get(second.id).id == second.id


# ---------------------------------------------------------------------------------------------
# REINFORCE


def test_reinforcement_from_a_second_source_makes_a_hidden_trait_visible(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "reinforce", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id,),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        derived_id = memory.consolidate(evidence_ids=(first.id,)).operations[0].created_ids[0]
        hidden = memory.get(derived_id)
        assert hidden.context is not None and hidden.context.visible is False
        assert not [hit for hit in memory.search("patient", limit=10) if hit.id == derived_id]

        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(second.id,),
                ),
            )
        )
        record = memory.consolidate(evidence_ids=(second.id, derived_id)).operations[0]

        assert record.changed_ids == (derived_id,)
        visible = memory.get(derived_id)
        assert visible.context is not None
        assert visible.context.visible is True
        assert set(visible.context.evidence_ids) == {first.id, second.id}
        assert visible.context.confidence == pytest.approx(1 - 0.4 * 0.4)
        assert [hit.id for hit in memory.search("patient", limit=10) if hit.id == derived_id]

        assert memory.rollback(record.operation_id) is True
        reverted = memory.get(derived_id)
        assert reverted.context is not None
        assert reverted.context.visible is False
        assert reverted.context.evidence_ids == (first.id,)
        assert reverted.context.confidence == pytest.approx(0.6)


def test_reinforcement_rejects_an_observation_target_and_an_existing_link(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "reinforce-bad", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id,),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        derived_id = memory.consolidate(evidence_ids=(first.id,)).operations[0].created_ids[0]

        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(first.id,),
                    evidence_ids=(second.id,),
                ),
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(first.id,),
                ),
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=("missing-derived-record",),
                    evidence_ids=(second.id,),
                ),
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(derived_id,),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id, derived_id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == [
            "not_derived",
            "already_linked",
            "unknown_target",
            "target_is_evidence",
        ]


def test_rollback_retires_only_the_evidence_the_operation_added(tmp_path: Path) -> None:
    """A consolidation onto an existing derived record must not un-link its earlier source."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "exact-undo", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        proposal = _trait("Ana", "patient")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id,),
                    proposal=proposal,
                ),
            )
        )
        derived_id = memory.consolidate(evidence_ids=(first.id,)).operations[0].created_ids[0]

        # A model-inferred trait derives the same identity from any evidence set, so this
        # re-proposal attaches a second source instead of creating a second record.
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=proposal,
                ),
            )
        )
        record = memory.consolidate(evidence_ids=(first.id, second.id)).operations[0]

        assert record.created_ids == ()
        assert record.changed_ids == (derived_id,)
        widened = memory.get(derived_id)
        assert widened.context is not None
        assert set(widened.context.evidence_ids) == {first.id, second.id}

        assert memory.rollback(record.operation_id) is True
        reverted = memory.get(derived_id)
        assert reverted.context is not None
        # Only the newly attached source is gone; the original support and record survive.
        assert reverted.context.evidence_ids == (first.id,)


def test_reinforcement_refuses_a_proposal_that_re_cites_an_existing_source(
    tmp_path: Path,
) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "restate", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id,),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        derived_id = memory.consolidate(evidence_ids=(first.id,)).operations[0].created_ids[0]
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(first.id, second.id),
                ),
            )
        )
        # `derived_id` is hidden, so it can only enter the target window by being named.
        report = memory.consolidate(evidence_ids=(first.id, second.id, derived_id))

        assert [reason for _operation, reason in report.rejected] == ["already_linked"]
        stable = memory.get(derived_id)
        assert stable.context is not None
        assert stable.context.evidence_ids == (first.id,)


# ---------------------------------------------------------------------------------------------
# CORRECT


def test_correction_retires_a_derived_inference_and_rollback_restores_it(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "correct", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        derived_id = (
            memory.consolidate(evidence_ids=(first.id, second.id)).operations[0].created_ids[0]
        )
        before = memory.get(derived_id)
        assert before.context is not None

        consolidator._scripts.append(
            (MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(derived_id,)),)
        )
        record = memory.consolidate(evidence_ids=(derived_id,)).operations[0]

        assert record.changed_ids == (derived_id,)
        assert not [hit for hit in memory.search("patient", limit=10) if hit.id == derived_id]
        # `get()` still exposes the retired version for audit.
        retired = memory.get(derived_id)
        assert retired.context is not None
        assert retired.context.retired_at is not None

        assert memory.rollback(record.operation_id) is True
        restored = memory.get(derived_id)
        assert restored.context is not None
        assert restored.context.retired_at is None
        assert restored.context.valid_from == before.context.valid_from
        assert restored.context.valid_until == before.context.valid_until
        assert [hit.id for hit in memory.search("patient", limit=10) if hit.id == derived_id]


def test_correction_refuses_a_raw_observation(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "correct-raw", consolidator) as memory:
        (source,) = _observations(memory, "Ana waited calmly")
        consolidator._scripts.append(
            (MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(source.id,)),)
        )
        report = memory.consolidate(evidence_ids=(source.id,))

        assert [reason for _operation, reason in report.rejected] == ["not_derived"]
        assert memory.get(source.id).context is not None


# ---------------------------------------------------------------------------------------------
# FORGET


def test_forgetting_leaves_search_but_stays_in_get_and_list(tmp_path: Path) -> None:
    with _memory(tmp_path / "forget") as memory:
        first, second = _observations(memory, "the kettle boiled", "the door closed")

        record = memory.forget((first.id,))

        assert record is not None
        assert record.operation.intent is MemoryIntent.FORGET
        assert record.model_id is None
        assert record.changed_ids == (first.id,)
        assert not [hit for hit in memory.search("kettle boiled", limit=10) if hit.id == first.id]
        assert [hit.id for hit in memory.search("door closed", limit=10) if hit.id == second.id]

        audited = memory.get(first.id)
        assert audited.forgotten_at == record.applied_at
        listed = {item.id: item.forgotten_at for item in memory.list().items}
        assert listed[first.id] == record.applied_at
        assert listed[second.id] is None

        assert memory.forget((first.id,)) is None
        assert memory.forget(()) is None
        # The host names IDs directly, so an unknown one is an error like `get()` and `delete()`,
        # and a set containing an already-forgotten record applies nothing at all.
        with pytest.raises(MemoryNotFoundError):
            memory.forget(("unknown-id",))
        with pytest.raises(MemoryNotFoundError):
            memory.forget((second.id, "unknown-id"))
        assert memory.get(second.id).forgotten_at is None
        assert memory.forget((first.id, second.id)) is None
        assert memory.get(second.id).forgotten_at is None

        assert memory.rollback(record.operation_id) is True
        assert memory.get(first.id).forgotten_at is None
        assert [hit.id for hit in memory.search("kettle boiled", limit=10) if hit.id == first.id]


def test_a_forgotten_record_is_never_shown_to_the_backend(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "forget-evidence", consolidator) as memory:
        first, second = _observations(memory, "the kettle boiled", "the door closed")
        memory.forget((first.id,))

        memory.consolidate(evidence_ids=(first.id, second.id))
        memory.consolidate(query="the kettle boiled")
        memory.consolidate()

        assert [shown for shown, _trigger in consolidator.calls] == [
            (second.id,),
            (second.id,),
            (second.id,),
        ]


# ---------------------------------------------------------------------------------------------
# Log


def test_operations_lists_newest_first_and_rollback_reports_unknown_ids(tmp_path: Path) -> None:
    with _memory(tmp_path / "log") as memory:
        first, second, third = _observations(memory, "one", "two", "three")
        oldest = memory.forget((first.id,))
        middle = memory.forget((second.id,))
        newest = memory.forget((third.id,))
        assert oldest is not None and middle is not None and newest is not None

        logged = memory.operations()

        assert [record.operation_id for record in logged] == [
            newest.operation_id,
            middle.operation_id,
            oldest.operation_id,
        ]
        assert [record.operation.target_ids[0] for record in logged] == [
            third.id,
            second.id,
            first.id,
        ]
        assert memory.operations(limit=1) == logged[:1]
        assert memory.rollback(newest.operation_id + 1000) is False
        with pytest.raises(ValidationError):
            memory.rollback(0)
        with pytest.raises(ValidationError):
            memory.operations(limit=0)


def test_the_trigger_reaches_the_backend_and_the_log(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "trigger", consolidator) as memory:
        (source,) = _observations(memory, "Ana waited calmly")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(source.id,),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        report = memory.consolidate(trigger=MemoryTrigger.CONTRADICTION)

        assert consolidator.calls[0][1] is MemoryTrigger.CONTRADICTION
        assert report.operations[0].trigger is MemoryTrigger.CONTRADICTION
        with pytest.raises(ValidationError):
            memory.consolidate(trigger="whenever")  # type: ignore[arg-type]


def test_a_failing_backend_maps_to_a_model_error(tmp_path: Path) -> None:
    class Broken(ScriptedConsolidator):
        def consolidate(
            self,
            evidence: Sequence[MemoryRecord],
            *,
            trigger: MemoryTrigger,
        ) -> tuple[MemoryOperation, ...]:
            raise RuntimeError("provider exploded")

    with _memory(tmp_path / "broken", Broken()) as memory:
        _observations(memory, "one")
        with pytest.raises(ModelError) as failure:
            memory.consolidate()
        assert failure.value.reason == "model_failed"
        assert failure.value.stage == "consolidate"

    class Sloppy(ScriptedConsolidator):
        def consolidate(
            self,
            evidence: Sequence[MemoryRecord],
            *,
            trigger: MemoryTrigger,
        ) -> tuple[MemoryOperation, ...]:
            return ["not an operation"]  # type: ignore[return-value]

    with _memory(tmp_path / "sloppy", Sloppy()) as memory:
        _observations(memory, "one")
        with pytest.raises(ModelError) as failure:
            memory.consolidate()
        assert failure.value.reason == "response_invalid"


def test_an_empty_store_never_calls_the_backend(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "empty", consolidator) as memory:
        assert memory.consolidate() == ConsolidationReport()
        assert consolidator.calls == []


# ---------------------------------------------------------------------------------------------
# Async parity


def test_async_memory_mirrors_the_control_plane(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()

    async def scenario() -> None:
        memory = AsyncMemory(
            tmp_path / "async",
            embedder=TinyEmbedder(),
            consolidator=consolidator,
            minimum_relevance=0,
        )
        try:
            first = await memory.add("Ana waited calmly", occurred_at=OCCURRED)
            second = await memory.add(
                "Ana waited again",
                occurred_at=OCCURRED + timedelta(minutes=1),
            )
            consolidator._scripts.append(
                (
                    MemoryOperation(
                        intent=MemoryIntent.CONSOLIDATE,
                        evidence_ids=(first.id, second.id),
                        proposal=_trait("Ana", "patient"),
                    ),
                )
            )
            report = await memory.consolidate(
                evidence_ids=(first.id, second.id),
                trigger=MemoryTrigger.IDLE,
            )
            assert len(report.operations) == 1
            assert report.operations[0].trigger is MemoryTrigger.IDLE

            forgotten = await memory.forget((first.id,))
            assert forgotten is not None
            assert (await memory.get(first.id)).forgotten_at is not None
            assert [record.operation_id for record in await memory.operations()] == [
                forgotten.operation_id,
                report.operations[0].operation_id,
            ]
            assert await memory.rollback(forgotten.operation_id) is True
            assert (await memory.get(first.id)).forgotten_at is None
        finally:
            await memory.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------------------------
# Consolidation forgetting


def test_consolidation_can_retire_the_detail_its_derived_record_replaces(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "consolidation-forgetting", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again, calmly")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    target_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        record = memory.consolidate(evidence_ids=(first.id, second.id)).operations[0]
        derived_id = record.created_ids[0]

        # One operation, not two: the log shows consolidation forgetting as a CONSOLIDATE row
        # carrying `forgotten_ids`, distinct from a FORGET intent and from `delete()`.
        assert record.operation.intent is MemoryIntent.CONSOLIDATE
        assert set(record.forgotten_ids) == {first.id, second.id}
        assert [row.operation.intent for row in memory.operations()] == [MemoryIntent.CONSOLIDATE]
        assert not [hit for hit in memory.search("Ana waited", limit=10) if hit.id == first.id]
        assert [hit.id for hit in memory.search("patient", limit=10) if hit.id == derived_id]
        # Lineage survives: the sources are still readable and still cited as evidence.
        assert memory.get(first.id).forgotten_at == record.applied_at
        derived = memory.get(derived_id)
        assert derived.context is not None
        assert set(derived.context.evidence_ids) == {first.id, second.id}

        assert memory.rollback(record.operation_id) is True
        assert memory.get(first.id).forgotten_at is None
        assert memory.get(second.id).forgotten_at is None
        with pytest.raises(MemoryNotFoundError):
            memory.get(derived_id)


def test_consolidation_cannot_retire_a_record_it_did_not_cite(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "retire-uncited", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "the door closed")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id,),
                    target_ids=(second.id,),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert [reason for _operation, reason in report.rejected] == ["target_not_evidence"]
        assert memory.get(second.id).forgotten_at is None
        assert memory.operations() == ()


def test_one_pass_may_not_contradict_itself(tmp_path: Path) -> None:
    """op1 consolidating from A and op2 forgetting A is the reachable form of a stale proposal."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "inconsistent", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
                MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id,)),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert len(report.operations) == 1
        assert [reason for _operation, reason in report.rejected] == ["inconsistent_batch"]
        assert memory.get(first.id).forgotten_at is None


def test_one_pass_may_not_build_on_evidence_it_just_retired(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "inconsistent-reverse", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id,)),
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert [record.operation.intent for record in report.operations] == [MemoryIntent.FORGET]
        assert [reason for _operation, reason in report.rejected] == ["inconsistent_batch"]


def test_a_correction_and_its_replacement_fit_in_one_pass(tmp_path: Path) -> None:
    """Update is CORRECT plus CONSOLIDATE in one batch; the consumed-set rule must allow it."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "update", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))
        wrong_id = report.operations[0].created_ids[0]

        consolidator._scripts.append(
            (
                MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(wrong_id,)),
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "impatient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id, wrong_id))

        assert report.rejected == ()
        assert [record.operation.intent for record in report.operations] == [
            MemoryIntent.CORRECT,
            MemoryIntent.CONSOLIDATE,
        ]
        replacement = memory.get(report.operations[1].created_ids[0])
        assert replacement.context is not None and replacement.context.value == "impatient"


# ---------------------------------------------------------------------------------------------
# Trust boundary


def test_a_backend_cannot_act_on_a_record_it_was_never_shown(tmp_path: Path) -> None:
    """A backend shown only A may not forget or correct an unrelated B; the host still may."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "unshown", consolidator) as memory:
        first, hidden = _observations(memory, "Ana waited calmly", "the door closed")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id,),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        derived_id = memory.consolidate(evidence_ids=(first.id,)).operations[0].created_ids[0]

        consolidator._scripts.append(
            (
                MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(hidden.id,)),
                MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(derived_id,)),
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(first.id,),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id,))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["target_not_shown"] * 3
        assert memory.get(hidden.id).forgotten_at is None
        assert [record.operation.intent for record in memory.operations()] == [
            MemoryIntent.CONSOLIDATE
        ]

        # The host names IDs directly and is the authority, so the same target still works.
        applied = memory.forget((hidden.id,))
        assert applied is not None and applied.forgotten_ids == (hidden.id,)


def test_a_multi_target_operation_applies_all_of_it_or_none(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "all-or-nothing", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        first_pass = memory.consolidate(evidence_ids=(first.id, second.id))
        derived_id = first_pass.operations[0].created_ids[0]

        consolidator._scripts.append(
            (
                # One derived target and one raw observation: the derived half used to be applied
                # on its own while the log row still listed both.
                MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(derived_id, first.id)),
                MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id, "no-such-id")),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id, derived_id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == [
            "not_derived",
            "unknown_target",
        ]
        assert memory.get(first.id).forgotten_at is None
        assert [record.operation.intent for record in memory.operations()] == [
            MemoryIntent.CONSOLIDATE
        ]

        # An already-forgotten record inside a larger FORGET refuses the whole operation too.
        assert memory.forget((second.id,)) is not None
        consolidator._scripts.append(
            (MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id, second.id)),)
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id, derived_id))

        assert [reason for _operation, reason in report.rejected] == ["already_forgotten"]
        assert memory.get(first.id).forgotten_at is None


def test_a_target_forgotten_between_validation_and_apply_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The apply transaction re-checks what validation read; a target that moved is refused."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "stale-target", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        original = LocalStore.apply_control_operation
        interleaved: list[str] = []

        def forget_first(
            self: LocalStore,
            operation: StoredOperation,
            **effects: object,
        ) -> StoredOperation | None:
            if not interleaved:
                interleaved.append(first.id)
                original(
                    self,
                    replace(operation, operation_key="interleaved-forget", intent="forget"),
                    forget_ids=(first.id,),
                )
            return cast("StoredOperation | None", cast(Any, original)(self, operation, **effects))

        monkeypatch.setattr(LocalStore, "apply_control_operation", forget_first)
        consolidator._scripts.append(
            (MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id, second.id)),)
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["stale"]
        # Only the interleaved write stands. The proposal applied nothing -- not even the half it
        # could still have applied -- and left no log row.
        assert memory.get(second.id).forgotten_at is None
        assert [row.operation.intent for row in memory.operations()] == [MemoryIntent.FORGET]
        assert memory.operations()[0].forgotten_ids == (first.id,)


def test_evidence_forgotten_between_validation_and_apply_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "stale-evidence", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")
        original = LocalStore.apply_formation
        interleaved: list[str] = []

        def forget_first(self: LocalStore, *args: object, **kwargs: object) -> bool:
            if not interleaved:
                interleaved.append(first.id)
                self.apply_control_operation(
                    StoredOperation(
                        operation_key="interleaved-forget",
                        intent="forget",
                        trigger="manual",
                        operation_json="{}",
                        applied_at=OCCURRED,
                    ),
                    forget_ids=(first.id,),
                )
            return cast(bool, cast(Any, original)(self, *args, **kwargs))

        monkeypatch.setattr(LocalStore, "apply_formation", forget_first)
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["stale"]
        assert [row.operation_key for row in memory._store.read_operations()] == [
            "interleaved-forget"
        ]


# ---------------------------------------------------------------------------------------------
# Durable triggers


class DispositionFormer:
    """Forms one model-inferred trait per observation, the way the fast path does."""

    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "former-test"
    formation_space = "former-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple((_trait("Ana", "patient"),) for _input in inputs)

    def close(self) -> None:
        pass


def test_formed_evidence_no_operation_has_weighed_is_a_durable_candidate(tmp_path: Path) -> None:
    with _memory(tmp_path / "due-evidence", former=DispositionFormer()) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again")

        due = memory.consolidation_candidates()
        evidence = [row for row in due if row.trigger is MemoryTrigger.EVIDENCE]

        assert len(evidence) == 1
        candidate = evidence[0]
        assert candidate.evidence_count == 2
        assert set(candidate.memory_ids) >= {first.id, second.id}
        assert candidate.memory_ids[0] not in {first.id, second.id}
        with pytest.raises(ValidationError):
            memory.consolidation_candidates(limit=0)


def test_a_forgotten_source_stops_supporting_the_candidate_it_formed(tmp_path: Path) -> None:
    with _memory(tmp_path / "forgotten-source", former=DispositionFormer()) as memory:
        (first,) = _observations(memory, "Ana waited calmly")
        assert [row for row in memory.consolidation_candidates() if first.id in row.memory_ids]

        memory.forget([first.id])

        # `consolidate()` reads the shown records with `active_only=True`, so a candidate naming a
        # forgotten source would ask the backend to weigh evidence it is never shown, and count it.
        assert [
            row
            for row in memory.consolidation_candidates()
            if row.trigger is MemoryTrigger.EVIDENCE and first.id in row.memory_ids
        ] == []


def test_a_deliberated_candidate_leaves_the_queue_until_new_evidence_arrives(
    tmp_path: Path,
) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "settled", consolidator, former=DispositionFormer()) as memory:
        (first,) = _observations(memory, "Ana waited calmly")
        candidate = memory.consolidation_candidates()[0]
        derived_id = candidate.memory_ids[0]
        assert first.id in candidate.memory_ids

        consolidator._scripts.append(
            (MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(derived_id,)),)
        )
        applied = memory.consolidate(
            evidence_ids=candidate.memory_ids,
            trigger=candidate.trigger,
        )

        assert len(applied.operations) == 1
        assert applied.operations[0].trigger is MemoryTrigger.EVIDENCE
        # The derived record's evidence now predates the operation that weighed it.
        assert [
            row for row in memory.consolidation_candidates() if row.memory_ids[0] == derived_id
        ] == []


def test_feedback_and_contradiction_are_derived_from_state_the_store_already_holds(
    tmp_path: Path,
) -> None:
    from mindbridge.context import _CONFLICT_KINDS
    from mindbridge.infrastructure.local.store import _CONFLICT_KINDS as _STORE_CONFLICT_KINDS

    # One infrastructure copy of the compiler's rule; they must not drift apart.
    assert set(_STORE_CONFLICT_KINDS) == {kind.value for kind in _CONFLICT_KINDS}

    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "due-feedback", consolidator) as memory:
        sources = _observations(
            memory,
            "Ana waited calmly",
            "Ana waited again, calmly",
            "Ana snapped at the delay",
            "Ana snapped again at the delay",
        )
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(sources[0].id, sources[1].id),
                    proposal=_trait("Ana", "patient"),
                ),
            )
        )
        memory.consolidate(evidence_ids=(sources[0].id, sources[1].id))
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(sources[2].id, sources[3].id),
                    proposal=_trait("Ana", "impatient"),
                ),
            )
        )
        memory.consolidate(evidence_ids=(sources[2].id, sources[3].id))
        memory.reinforce((sources[0].id,))

        due = memory.consolidation_candidates()
        by_trigger = {row.trigger: row for row in due}

        contradiction = by_trigger[MemoryTrigger.CONTRADICTION]
        assert contradiction.evidence_count == 2
        assert len(contradiction.memory_ids) == 2
        feedback = by_trigger[MemoryTrigger.FEEDBACK]
        assert feedback.memory_ids == (sources[0].id,)
        assert feedback.evidence_count == 1


def test_a_concurrent_duplicate_is_refused_inside_the_transaction(tmp_path: Path) -> None:
    """The caller's pre-check can lose a race; the write path must still refuse, not raise."""
    applied_at = datetime(2026, 3, 2, tzinfo=timezone.utc)
    pending = StoredOperation(
        operation_key="shared-key",
        intent="consolidate",
        trigger="evidence",
        operation_json="{}",
        applied_at=applied_at,
    )
    with LocalStore(tmp_path / "race") as store:
        assert (
            store.apply_formation(
                (),
                (),
                evidence=(),
                source_memory_ids=(),
                recipe="consolidator-test:v1",
                completed_at=applied_at,
                operation=pending,
            )
            is True
        )
        assert (
            store.apply_formation(
                (),
                (),
                evidence=(),
                source_memory_ids=(),
                recipe="consolidator-test:v1",
                completed_at=applied_at,
                operation=replace(pending, model_id="other"),
            )
            is False
        )
        assert len(store.read_operations()) == 1


# ---------------------------------------------------------------------------------------------
# Replay


class Replayer:
    """Replays whatever the log recorded for the evidence set it is shown."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self, logged: Sequence[MemoryOperationRecord]) -> None:
        self._logged = tuple(logged)

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        shown = {record.id for record in evidence}
        return tuple(
            row.operation for row in self._logged if set(row.operation.evidence_ids) <= shown
        )

    def close(self) -> None:
        pass


def test_a_logged_operation_sequence_replays_against_a_fresh_store(tmp_path: Path) -> None:
    """Gate 3's replay item: the log alone is enough to reproduce the derived state.

    `applied_at` is re-stamped at replay, so transaction times differ by construction. What the
    log is supposed to determine -- the operation keys, the derived IDs, and the evidence
    lineage -- does not depend on it.
    """
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "origin", consolidator) as origin:
        first, second, third = _observations(
            origin,
            "Ana waited calmly",
            "Ana waited again, calmly",
            "Ana waited a third time",
        )
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(first.id, second.id),
                    target_ids=(second.id,),
                    proposal=_trait("Ana", "patient"),
                    rationale="two independent waits",
                ),
            )
        )
        origin.consolidate(evidence_ids=(first.id, second.id))
        logged = tuple(reversed(origin.operations()))
        recorded = [
            (row.operation.intent, row.created_ids, row.changed_ids, row.forgotten_ids)
            for row in logged
        ]
        keys = [operation_key(row.operation, recipe=row.recipe) for row in logged]

    with _memory(tmp_path / "replayed", Replayer(logged)) as replayed:
        sources = _observations(
            replayed,
            "Ana waited calmly",
            "Ana waited again, calmly",
            "Ana waited a third time",
        )
        # Content-addressed identity: the same observations get the same IDs in a fresh store.
        assert [record.id for record in sources] == [first.id, second.id, third.id]
        report = replayed.consolidate(evidence_ids=(first.id, second.id))

        assert report.rejected == ()
        replayed_log = tuple(reversed(replayed.operations()))
        assert [
            (row.operation.intent, row.created_ids, row.changed_ids, row.forgotten_ids)
            for row in replayed_log
        ] == recorded
        assert [operation_key(row.operation, recipe=row.recipe) for row in replayed_log] == keys
        derived = replayed.get(report.operations[0].created_ids[0])
        assert derived.context is not None
        assert set(derived.context.evidence_ids) == {first.id, second.id}
        assert replayed.get(second.id).forgotten_at is not None
