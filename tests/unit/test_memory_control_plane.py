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

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AssetRef,
    AsyncMemory,
    Blob,
    ConsolidationBackend,
    ConsolidationReport,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationProposal,
    IdentityClaim,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryNotFoundError,
    MemoryOperation,
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
        report = memory.consolidate(evidence_ids=(first.id, second.id))

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
        assert memory.forget(("unknown-id",)) is None
        assert memory.forget(()) is None

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
            rejected = memory.consolidate(evidence_ids=(record.id,)).rejected
            assert [reason for _operation, reason in rejected] == ["naming_assertion"]

        # `forget()` is the same kernel path under host authority, so it refuses too.
        assert memory.forget((assertion,)) is None

        # The invariant every accepted operation has to leave standing.
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) == "Li"
        assert memory.operations()[0].operation.intent is MemoryIntent.IDENTIFY
