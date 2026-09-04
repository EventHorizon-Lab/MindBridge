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
    AssetRef,
    AsyncMemory,
    Blob,
    ConsolidationBackend,
    ConsolidationReport,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationInput,
    FormationProposal,
    IdentityClaim,
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
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
    ValidationError,
)
from mindbridge.control import dump_operation, load_operation, operation_key
from mindbridge.infrastructure.local import LocalStore
from mindbridge.infrastructure.local.store import StoredOperation

OCCURRED = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)


class OnePersonSpeech:
    """Hears the same single speaker in every clip, so one identity owns every segment."""

    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "one-person-speech"
    transcription_space = "one-person-speech:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        return tuple(
            SpeechAnalysis(
                turns=(SpeechTurn(0, 900, "I am Li, I live next door", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            )
            for _asset in assets
        )

    def close(self) -> None:
        pass


class OnePersonFace:
    """Sees the same single face in every clip, so it corroborates with the voice."""

    face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    face_model = "one-person-face"
    face_space = "one-person-face:2:test"
    face_analysis_space = "one-person-face-analysis:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis((FaceEmbedding("face-0", (0.0, 1.0), (0.1, 0.1, 0.4, 0.5), None),))
            for _asset in assets
        )

    def close(self) -> None:
        pass


def _identity_memory(tmp_path: Path, consolidator: object) -> Memory:
    return Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        transcriber=OnePersonSpeech(),
        face_analyzer=OnePersonFace(),
        consolidator=consolidator,  # type: ignore[arg-type]
        identity_link_min_assets=1,
        minimum_relevance=0,
    )


def _asserted_name(memory: Memory, identity_id: str) -> str | None:
    """Return the name the currently visible naming assertion of one identity claims."""
    names = [
        record.context.subject
        for record in memory.list(limit=100).items
        if record.context is not None
        and record.context.kind is MemoryKind.ENTITY
        and record.context.identity_id == identity_id
        and record.context.visible
        and record.context.retired_at is None
    ]
    assert len(names) <= 1
    return names[0] if names else None


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
    claim = IdentityClaim(identity_id="identity_1", name="Li")
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
        # IDENTIFY carries a claim and nothing else a kernel would have to reconcile with it.
        {"intent": MemoryIntent.IDENTIFY},
        {"intent": MemoryIntent.IDENTIFY, "evidence_ids": ("a",)},
        {"intent": MemoryIntent.IDENTIFY, "claim": claim, "target_ids": ("t",)},
        {"intent": MemoryIntent.IDENTIFY, "claim": claim, "proposal": proposal},
        {"intent": MemoryIntent.IDENTIFY, "claim": "Li"},
        {
            "intent": MemoryIntent.CONSOLIDATE,
            "evidence_ids": ("a",),
            "proposal": proposal,
            "claim": claim,
        },
        {"intent": MemoryIntent.FORGET, "target_ids": ("t",), "claim": claim},
    )
    for operation in malformed:
        with pytest.raises(ValidationError):
            MemoryOperation(**operation)  # type: ignore[arg-type]

    # A host naming somebody cites nothing, so an empty evidence set is well formed.
    assert MemoryOperation(intent=MemoryIntent.IDENTIFY, claim=claim).evidence_ids == ()
    assert MemoryOperation(
        intent=MemoryIntent.IDENTIFY, claim=claim, evidence_ids=("a", "a")
    ).evidence_ids == ("a",)
    with pytest.raises(ValidationError):
        IdentityClaim(identity_id="identity_1", name="bad\nname")
    with pytest.raises(ValidationError):
        IdentityClaim(identity_id=" ", name="Li")


def test_a_naming_claim_round_trips_through_the_log_and_keys_the_operation() -> None:
    claim = IdentityClaim(identity_id="identity_1", name="Li", relationship="neighbour")
    operation = MemoryOperation(
        intent=MemoryIntent.IDENTIFY,
        claim=claim,
        evidence_ids=("memory_1",),
        rationale="the stranger introduced themselves",
    )

    assert load_operation(dump_operation(operation)) == operation

    key = operation_key(operation, recipe="r")
    # Prose is logged and never interpreted, so it stays out of the idempotency identity; the
    # claim is the operation, so every part of it stays in.
    assert operation_key(replace(operation, rationale="reworded"), recipe="r") == key
    assert operation_key(replace(operation, claim=replace(claim, name="Li Hua")), recipe="r") != key
    assert (
        operation_key(replace(operation, claim=replace(claim, relationship=None)), recipe="r")
        != key
    )


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
# Naming as a proposal


def test_deleting_every_cited_source_takes_the_agent_asserted_name_with_it(
    tmp_path: Path,
) -> None:
    """Losing evidence un-projects an inferred name; losing all of it removes the assertion.

    An agent's IDENTIFY links its assertion to the memories it cited, so deleting one source
    drops the name below the two-group threshold and deleting the last one cascade-deletes the
    assertion itself. Reprojecting only the record the caller named left the registry and the
    search index answering to a name no assertion supported any more, which is the
    audit-versus-registry contradiction this round exists to remove.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "cascade", consolidator) as memory:
        first = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        second = memory.add(Blob(b"stranger speaks again", "video/mp4", "two.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id

        def identify(*evidence: str) -> MemoryOperation:
            return MemoryOperation(
                intent=MemoryIntent.IDENTIFY,
                claim=IdentityClaim(identity_id=identity_id, name="Li"),
                evidence_ids=evidence,
                rationale="the stranger said so",
            )

        for cited in (first.id, second.id):
            consolidator._scripts.append((identify(cited),))
            assert not memory.consolidate(evidence_ids=(first.id, second.id)).rejected
        profile = memory.identity(identity_id)
        assert profile is not None and profile.name == "Li"

        # Losing one source drops an inferred name below the two-group threshold, so it stops
        # being projected while the assertion itself survives on its remaining evidence.
        assert memory.delete(first.id) is True
        assert _asserted_name(memory, identity_id) is None
        profile = memory.identity(identity_id)
        assert profile is not None and profile.name is None

        # The second delete removes the assertion's last evidence, so the assertion goes too.
        assert memory.delete(second.id) is True
        assert not [
            item
            for item in memory.list().items
            if item.context is not None and item.context.identity_id == identity_id
        ]
        profile = memory.identity(identity_id)
        assert profile is not None and profile.name is None
        assert profile.confirmed is False


def test_an_agent_names_a_person_only_from_evidence_that_contains_them(tmp_path: Path) -> None:
    """Scenario step 3: the stranger introduces themselves and the agent proposes the name.

    The claim is a proposal like any other, so the kernel checks that the person exists and that
    the cited evidence actually contains them, and keeps an inferred name out of the projection
    until two independent evidence groups support it.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "identify", consolidator) as memory:
        first = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id
        assert identity_id == memory.speech(first.id)[0].speaker_id
        second = memory.add(Blob(b"stranger speaks again", "video/mp4", "two.mp4"))
        elsewhere = memory.add("the courier left a parcel", occurred_at=OCCURRED)

        def identify(*evidence: str, identity: str = identity_id) -> MemoryOperation:
            return MemoryOperation(
                intent=MemoryIntent.IDENTIFY,
                claim=IdentityClaim(identity_id=identity, name="Li", relationship="neighbour"),
                evidence_ids=evidence,
                rationale="the stranger said so",
            )

        consolidator._scripts.append((identify(first.id, identity="identity_missing"),))
        rejected = memory.consolidate(evidence_ids=(first.id,)).rejected
        assert [reason for _operation, reason in rejected] == ["unknown_identity"]

        # A name pinned on somebody the cited evidence never contained is the failure the
        # involvement check exists to make impossible.
        consolidator._scripts.append((identify(elsewhere.id),))
        rejected = memory.consolidate(evidence_ids=(elsewhere.id,)).rejected
        assert [reason for _operation, reason in rejected] == ["identity_not_in_evidence"]

        consolidator._scripts.append((identify(),))
        rejected = memory.consolidate(evidence_ids=(first.id,)).rejected
        assert [reason for _operation, reason in rejected] == ["identity_not_in_evidence"]

        # One group of evidence logs the assertion but leaves the person unnamed.
        consolidator._scripts.append((identify(first.id),))
        report = memory.consolidate(evidence_ids=(first.id,))
        assert report.rejected == ()
        assert len(report.operations) == 1
        assert report.operations[0].operation.intent is MemoryIntent.IDENTIFY
        assert report.operations[0].operation.claim is not None
        assert report.operations[0].operation.claim.name == "Li"
        assert memory.identity(identity_id) is not None
        assert memory.identity(identity_id).name is None  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) is None

        # A second independent group corroborates it, and the projection follows.
        consolidator._scripts.append((identify(second.id),))
        report = memory.consolidate(evidence_ids=(second.id,))
        assert report.rejected == ()
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) == "Li"
        assert [record.operation.intent for record in memory.operations()] == [
            MemoryIntent.IDENTIFY,
            MemoryIntent.IDENTIFY,
        ]

        # Reversing the corroborating operation takes the projected name with it.
        assert memory.rollback(memory.operations()[0].operation_id) is True
        assert memory.identity(identity_id).name is None  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) is None


def test_control_operations_refuse_to_touch_a_bound_naming_assertion(tmp_path: Path) -> None:
    """A name is not an inference to reinforce, correct, or forget behind the projection.

    Retiring the assertion through one of those intents left `identity()` and the indexed
    speech text answering to a name nothing asserted any more. `rollback()` of the operation
    that named the person is the reversal that recomputes both.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "guarded", consolidator) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id
        memory.register_identity(identity_id, "Li", relationship="neighbour")
        assertion = next(
            item.id
            for item in memory.list(limit=100).items
            if item.context is not None and item.context.identity_id == identity_id
        )

        scripted = (
            MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(assertion,)),
            MemoryOperation(
                intent=MemoryIntent.REINFORCE,
                target_ids=(assertion,),
                evidence_ids=(record.id,),
            ),
            MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(assertion,)),
        )
        for operation in scripted:
            consolidator._scripts.append((operation,))
            # The assertion is named in `evidence_ids`, so it is inside the window the kernel
            # gathered: the refusal below is the naming guard, not the window check that would
            # otherwise refuse a target the backend was never shown.
            rejected = memory.consolidate(evidence_ids=(record.id, assertion)).rejected
            assert [reason for _operation, reason in rejected] == ["naming_assertion"]

        # `forget()` is the same kernel path under host authority, so it refuses too.
        assert memory.forget((assertion,)) is None

        # The invariant every accepted operation has to leave standing.
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) == "Li"
        assert memory.operations()[0].operation.intent is MemoryIntent.IDENTIFY


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


# ---------------------------------------------------------------------------------------------
# Lineage supersession


def _state(value: str, *, valid_from: datetime) -> FormationProposal:
    """A STATE proposal: the kind whose lineage rule supersedes the claim it replaces."""
    return FormationProposal(
        kind=MemoryKind.STATE,
        content=f"Ana is in the {value}",
        subject="Ana",
        predicate="location",
        value=value,
        confidence=0.9,
        valid_from=valid_from,
    )


def _consolidate_state(
    memory: Memory,
    consolidator: ScriptedConsolidator,
    source: MemoryRecord,
    value: str,
) -> MemoryOperationRecord:
    consolidator._scripts.append(
        (
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=_state(value, valid_from=OCCURRED),
            ),
        )
    )
    return memory.consolidate(evidence_ids=(source.id,)).operations[0]


def test_lineage_supersession_is_logged_and_reversed(tmp_path: Path) -> None:
    """The kernel's own lineage rule reaches records the backend was never shown.

    That is deliberate -- it is a deterministic kernel effect, not a backend choice -- but it
    used to be invisible in the log and irreversible, so `rollback()` left the lineage with the
    superseded claim still retired and nothing standing in its place.
    """
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "supersession", consolidator) as memory:
        first, second = _observations(memory, "Ana is in the kitchen", "Ana moved to the garden")
        kitchen = _consolidate_state(memory, consolidator, first, "kitchen")
        kitchen_id = kitchen.created_ids[0]
        assert kitchen.superseded == ()

        garden = _consolidate_state(memory, consolidator, second, "garden")

        # The second pass only ever saw `second`; the record it superseded was never shown.
        assert consolidator.calls[-1][0] == (second.id,)
        assert garden.superseded == ((kitchen_id, 1),)
        retired = memory.get(kitchen_id)
        assert retired.context is not None and retired.context.retired_at is not None
        assert not [hit for hit in memory.search("kitchen", limit=10) if hit.id == kitchen_id]

        assert memory.rollback(garden.operation_id) is True
        restored = memory.get(kitchen_id)
        assert restored.context is not None and restored.context.retired_at is None
        assert [hit.id for hit in memory.search("kitchen", limit=10) if hit.id == kitchen_id]
        with pytest.raises(MemoryNotFoundError):
            memory.get(garden.created_ids[0])
        assert memory.operations()[0].rolled_back_at is not None


def test_operations_on_one_lineage_reverse_newest_first(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "supersession-order", consolidator) as memory:
        first, second, third = _observations(
            memory,
            "Ana is in the kitchen",
            "Ana moved to the garden",
            "Ana moved to the hall",
        )
        kitchen = _consolidate_state(memory, consolidator, first, "kitchen")
        garden = _consolidate_state(memory, consolidator, second, "garden")
        hall = _consolidate_state(memory, consolidator, third, "hall")
        assert garden.superseded == ((kitchen.created_ids[0], 1),)
        assert hall.superseded == ((garden.created_ids[0], 1),)

        # Reversing the middle operation first would restore the kitchen claim beside the hall
        # claim and delete the record the hall claim supersedes. Refused, and nothing moves.
        assert memory.rollback(garden.operation_id) is False
        assert memory.get(kitchen.created_ids[0]).context is not None
        assert memory.get(kitchen.created_ids[0]).context.retired_at is not None  # type: ignore[union-attr]
        assert memory.get(garden.created_ids[0]) is not None
        logged = {row.operation_id: row.rolled_back_at for row in memory.operations()}
        assert logged[garden.operation_id] is None

        assert memory.rollback(hall.operation_id) is True
        current = memory.get(garden.created_ids[0])
        assert current.context is not None and current.context.retired_at is None

        assert memory.rollback(garden.operation_id) is True
        oldest = memory.get(kitchen.created_ids[0])
        assert oldest.context is not None and oldest.context.retired_at is None
        assert [
            hit.id for hit in memory.search("kitchen", limit=10) if hit.id == kitchen.created_ids[0]
        ]


def test_reinforcement_refuses_a_target_a_correction_already_retired(tmp_path: Path) -> None:
    """A retired claim is not a standing derived claim, in this pass or an earlier one."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "reinforce-retired", consolidator) as memory:
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

        # One pass. The CORRECT commits first, so the REINFORCE behind it names a claim this
        # very pass withdrew; the batch guard used to compare only evidence against retirals.
        consolidator._scripts.append(
            (
                MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(derived_id,)),
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(second.id,),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(first.id, second.id, derived_id))

        assert [row.operation.intent for row in report.operations] == [MemoryIntent.CORRECT]
        assert [reason for _operation, reason in report.rejected] == ["inconsistent_batch"]

        # A later pass cannot see the retiral in a batch, so the target itself is checked.
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(second.id,),
                ),
            )
        )
        later = memory.consolidate(evidence_ids=(first.id, second.id, derived_id))

        assert later.operations == ()
        assert [reason for _operation, reason in later.rejected] == ["not_derived"]
        unchanged = memory.get(derived_id)
        assert unchanged.context is not None
        assert unchanged.context.evidence_ids == (first.id,)


def test_a_reinforce_target_corrected_between_validation_and_apply_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The version check belongs to REINFORCE, not to every caller of the apply transaction."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "stale-retired", consolidator) as memory:
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
        original = LocalStore.apply_control_operation
        interleaved: list[str] = []

        def correct_first(
            self: LocalStore,
            operation: StoredOperation,
            **effects: object,
        ) -> StoredOperation | None:
            if not interleaved:
                interleaved.append(derived_id)
                original(
                    self,
                    replace(operation, operation_key="interleaved-correct", intent="correct"),
                    correct_ids=(derived_id,),
                )
            return cast("StoredOperation | None", cast(Any, original)(self, operation, **effects))

        monkeypatch.setattr(LocalStore, "apply_control_operation", correct_first)
        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(derived_id,),
                    evidence_ids=(second.id,),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(second.id, derived_id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["stale"]
        assert memory._store.read_operations()[0].operation_key == "interleaved-correct"
        # Two rows: the first consolidation and the interleaved correction. The refused
        # reinforcement wrote nothing at all.
        assert len(memory.operations()) == 2
        supported = memory.get(derived_id)
        assert supported.context is not None
        assert supported.context.evidence_ids == (first.id,)
        # The host may still forget an already-corrected record: `require_active` stays loose.
        assert memory.forget((derived_id,)) is not None


def test_a_query_gathered_window_is_never_widened(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "query-window", consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "the kettle boiled")
        gathered = memory.search("Ana waited calmly", limit=1)[0].id
        outside = second.id if gathered == first.id else first.id
        consolidator._scripts.append(
            (MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(outside,)),)
        )
        report = memory.consolidate(query="Ana waited calmly", limit=1)

        # A query gathers the shown set and nothing else: with no host `evidence_ids`, the window
        # is exactly what the backend saw.
        assert [shown for shown, _trigger in consolidator.calls] == [(gathered,)]
        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["target_not_shown"]
        assert memory.get(outside).forgotten_at is None
        assert memory.operations() == ()


def test_consolidation_forgetting_refuses_a_bound_naming_assertion(tmp_path: Path) -> None:
    """The naming guard covers CONSOLIDATE too, whose own forgetting names targets of its own."""
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "consolidated", consolidator) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id
        memory.register_identity(identity_id, "Li", relationship="neighbour")
        assertion = next(
            item.id
            for item in memory.list(limit=100).items
            if item.context is not None and item.context.identity_id == identity_id
        )

        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(record.id, assertion),
                    target_ids=(assertion,),
                    proposal=_trait("Li", "patient"),
                ),
            )
        )
        report = memory.consolidate(evidence_ids=(record.id, assertion))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["naming_assertion"]
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) == "Li"
        assert [item.operation.intent for item in memory.operations()] == [MemoryIntent.IDENTIFY]


def test_every_operation_payload_field_is_derived_from_the_dataclass() -> None:
    """A new proposal field must reach the idempotency key, or two proposals hash alike."""
    import dataclasses

    from mindbridge import SpatialAnchor, SpatialContext
    from mindbridge.control import _proposal_payload, _spatial_payload

    spatial = SpatialContext(frame_id="home", anchor=SpatialAnchor.OBSERVER, x=1.0, y=2.0)
    proposal = replace(_trait("Li", "patient"), spatial=spatial)

    assert {field.name for field in dataclasses.fields(proposal)} == set(
        _proposal_payload(proposal)
    )
    assert {field.name for field in dataclasses.fields(spatial)} == set(_spatial_payload(spatial))


def test_renaming_a_person_back_restores_the_name_they_had(tmp_path: Path) -> None:
    """A name is a claim, and re-asserting one the host retracted has to land.

    The claim's memory ID is a function of the claim, so renaming back reaches a record that
    already exists with every version retired. Treating that as "already stored" made the
    rename a silent no-op: the registry kept the name the host had just replaced.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "rename-back", consolidator) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id

        memory.register_identity(identity_id, "Li")
        memory.register_identity(identity_id, "Li Hua")
        assert _asserted_name(memory, identity_id) == "Li Hua"

        memory.register_identity(identity_id, "Li")
        assert _asserted_name(memory, identity_id) == "Li"
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]


def test_registering_a_name_again_after_rolling_it_back_names_the_person(tmp_path: Path) -> None:
    """Rolling a registration back retracts the claim; making it again must re-assert it.

    The rollback leaves the record in place with its only version retired, and the log row
    marked rolled back. The repeat registration is therefore a fresh operation and has to
    produce a fresh version, not log an `identify` row over an identity that stays unnamed.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "re-register", consolidator) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id

        memory.register_identity(identity_id, "Li")
        assert memory.rollback(memory.operations()[0].operation_id) is True
        assert _asserted_name(memory, identity_id) is None

        memory.register_identity(identity_id, "Li")
        assert _asserted_name(memory, identity_id) == "Li"
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]
        # Both attempts are auditable, and exactly one of them still stands.
        rows = memory.operations()
        assert [row.operation.intent for row in rows] == [
            MemoryIntent.IDENTIFY,
            MemoryIntent.IDENTIFY,
        ]
        assert [row.rolled_back_at is None for row in rows].count(True) == 1


def test_an_unconfirmed_agent_name_does_not_erase_the_one_the_host_registered(
    tmp_path: Path,
) -> None:
    """A claim nobody can see must not be what displaces the standing one.

    The agent's single-evidence IDENTIFY is hidden until a second group corroborates it, so it
    accumulates beside the host's assertion the way a hidden inferred trait does. Retiring the
    predecessor before deciding visibility left the person nameless on one guess.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "unconfirmed", consolidator) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id
        memory.register_identity(identity_id, "Alice")

        consolidator._scripts.append(
            (
                MemoryOperation(
                    intent=MemoryIntent.IDENTIFY,
                    claim=IdentityClaim(identity_id=identity_id, name="Ana"),
                    evidence_ids=(record.id,),
                    rationale="the stranger said so",
                ),
            )
        )
        assert not memory.consolidate(evidence_ids=(record.id,)).rejected

        assert _asserted_name(memory, identity_id) == "Alice"
        assert memory.identity(identity_id).name == "Alice"  # type: ignore[union-attr]


def test_a_corroborated_agent_name_still_loses_to_the_one_the_host_stated(tmp_path: Path) -> None:
    """What the host said outranks what a model worked out, for names as for traits.

    Two independent evidence groups clear the corroboration bar, but the suppression rule that
    keeps an inference from contradicting an explicit user statement was written for TRAIT only
    and never fired for the ENTITY assertions it now also gates.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "outranked", consolidator) as memory:
        first = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        second = memory.add(Blob(b"stranger speaks again", "video/mp4", "two.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id
        memory.register_identity(identity_id, "Alice")

        for cited in (first.id, second.id):
            consolidator._scripts.append(
                (
                    MemoryOperation(
                        intent=MemoryIntent.IDENTIFY,
                        claim=IdentityClaim(identity_id=identity_id, name="Ana"),
                        evidence_ids=(cited,),
                        rationale="the stranger said so",
                    ),
                )
            )
            assert not memory.consolidate(evidence_ids=(first.id, second.id)).rejected

        assert _asserted_name(memory, identity_id) == "Alice"
        assert memory.identity(identity_id).name == "Alice"  # type: ignore[union-attr]


def test_deleting_one_source_of_an_agent_asserted_name_takes_it_out_of_indexed_speech(
    tmp_path: Path,
) -> None:
    """Un-projecting a name has to repaint the transcripts that quoted it, not just the registry.

    Deleting a memory an assertion cites is not deleting the assertion, so the delete path saw no
    naming record and skipped the reindex: `identity()` reported nobody while the stored text and
    its vectors still answered to the name.
    """
    consolidator = ScriptedConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        transcriber=OnePersonSpeech(),
        face_analyzer=OnePersonFace(),
        consolidator=consolidator,
        identity_link_min_assets=1,
        index_speech=True,
        minimum_relevance=0,
    ) as memory:
        first = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        second = memory.add(Blob(b"stranger speaks again", "video/mp4", "two.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id

        def identify(*evidence: str) -> MemoryOperation:
            return MemoryOperation(
                intent=MemoryIntent.IDENTIFY,
                claim=IdentityClaim(identity_id=identity_id, name="Li"),
                evidence_ids=evidence,
                rationale="the stranger said so",
            )

        for cited in (first.id, second.id):
            consolidator._scripts.append((identify(cited),))
            assert not memory.consolidate(evidence_ids=(first.id, second.id)).rejected
        assert '"speaker_name":"Li"' in memory.get(second.id).content

        # One group left is below the threshold for an inferred name, so it stops being projected
        # and the surviving transcript has to stop carrying it in the same commit.
        assert memory.delete(first.id) is True

        profile = memory.identity(identity_id)
        assert profile is not None and profile.name is None
        assert '"speaker_name":"Li"' not in memory.get(second.id).content
