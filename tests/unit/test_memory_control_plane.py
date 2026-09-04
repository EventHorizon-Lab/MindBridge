"""Kernel contract for the agentic memory control plane.

Every test drives the public SDK. The backend here only proposes; what these assert is that the
kernel validates each proposal, applies exactly the effect its intent allows, logs it once, and
can reverse it.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    AsyncMemory,
    Blob,
    ConsolidationBackend,
    ConsolidationCandidate,
    ConsolidationReport,
    ContextBudget,
    DeliberationReport,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationInput,
    FormationProposal,
    IdentityChange,
    IdentityClaim,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryPlugins,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    RetrievalScope,
    SearchHit,
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


def _observations(
    memory: Memory,
    *contents: str,
    places: Sequence[str | None] | None = None,
    tags: Sequence[Mapping[str, object] | None] | None = None,
) -> tuple[MemoryRecord, ...]:
    """Add one observation per content, optionally placing and tagging them pairwise."""
    return tuple(
        memory.add(
            content,
            occurred_at=OCCURRED + timedelta(minutes=index),
            context=ObservationContext(
                basis=EvidenceBasis.OBSERVATION,
                source_id=f"cam-{index}",
                place_id=place,
            ),
            metadata=tag,
        )
        for index, (content, place, tag) in enumerate(
            zip(
                contents,
                places or (None,) * len(contents),
                tags or (None,) * len(contents),
                strict=True,
            )
        )
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
            MemoryIntent.MERGE,
        ]

        # Reversing the corroborating operation takes the projected name with it.
        assert memory.rollback(memory.operations()[0].operation_id) is True
        assert memory.identity(identity_id).name is None  # type: ignore[union-attr]
        assert _asserted_name(memory, identity_id) is None


def test_identify_creation_cannot_be_rolled_back_before_its_later_reinforcement(
    tmp_path: Path,
) -> None:
    """A later active operation depends on the deterministic assertion the first one created."""
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "identify-order", consolidator) as memory:
        first = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        second = memory.add(Blob(b"stranger returns", "video/mp4", "two.mp4"))
        identity_id = memory.faces(first.id)[0].identity_id

        def identify(source_id: str) -> MemoryOperation:
            return MemoryOperation(
                intent=MemoryIntent.IDENTIFY,
                claim=IdentityClaim(identity_id=identity_id, name="Li"),
                evidence_ids=(source_id,),
                rationale=f"evidence from {source_id}",
            )

        operations = []
        for source in (first, second):
            consolidator._scripts.append((identify(source.id),))
            operations.append(memory.consolidate(evidence_ids=(source.id,)).operations[0])

        # The second operation changes the same deterministic assertion, so deleting/retracting
        # the first operation's output would strand that still-active log row.
        assert memory.rollback(operations[0].operation_id) is False
        assert memory.identity(identity_id).name == "Li"  # type: ignore[union-attr]

        # Newest-first succeeds. Once both evidence-adding operations are gone, the first
        # assertion is retired and no evidence link from its creation survives.
        assert memory.rollback(operations[1].operation_id) is True
        assert memory.rollback(operations[0].operation_id) is True
        assertion = next(
            item
            for item in memory.list(limit=100).items
            if item.context is not None and item.context.identity_id == identity_id
        )
        assert assertion.context is not None
        assert assertion.context.retired_at is not None
        assert assertion.context.evidence_ids == ()


def test_a_historical_name_can_become_current_again_and_be_rolled_back(
    tmp_path: Path,
) -> None:
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path / "rename-back", consolidator) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id

        memory.register_identity(identity_id, "Alice")
        memory.register_identity(identity_id, "Bob")
        memory.register_identity(identity_id, "Alice")

        assert memory.identity(identity_id).name == "Alice"  # type: ignore[union-attr]
        newest = memory.operations()[0]
        assert newest.operation.claim is not None and newest.operation.claim.name == "Alice"
        assert memory.rollback(newest.operation_id) is True
        assert memory.identity(identity_id).name == "Bob"  # type: ignore[union-attr]


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


def test_a_logged_operation_sequence_replays_against_a_fresh_store(tmp_path: Path) -> None:
    """Gate 3's replay item: the log alone is enough to reproduce the derived state.

    Replay is the public `apply()` surface, not a backend impersonating one: the fresh store is
    configured with the same consolidation recipe -- a derived representation belongs to a
    recipe -- but its backend is never called, and each logged `MemoryOperation` goes through the
    same kernel validation a proposal does. `applied_at` is re-stamped at replay, so transaction
    times differ by construction. What the log is supposed to determine -- the operation keys,
    the derived IDs, and the evidence lineage -- does not depend on it.
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

    silent = SilentConsolidator()
    with _memory(tmp_path / "replayed", silent) as replayed:
        sources = _observations(
            replayed,
            "Ana waited calmly",
            "Ana waited again, calmly",
            "Ana waited a third time",
        )
        # Content-addressed identity: the same observations get the same IDs in a fresh store.
        assert [record.id for record in sources] == [first.id, second.id, third.id]
        applied = tuple(replayed.apply(row.operation) for row in logged)
        # Replay reasons about nothing: the configured backend was never asked.
        assert silent.calls == 0

        replayed_log = tuple(reversed(replayed.operations()))
        assert [
            (row.operation.intent, row.created_ids, row.changed_ids, row.forgotten_ids)
            for row in replayed_log
        ] == recorded
        assert [operation_key(row.operation, recipe=row.recipe) for row in replayed_log] == keys
        derived = replayed.get(applied[0].created_ids[0])
        assert derived.context is not None
        assert set(derived.context.evidence_ids) == {first.id, second.id}
        assert replayed.get(second.id).forgotten_at is not None

        # The same operation twice is one operation, exactly as it is for a proposal.
        with pytest.raises(ValidationError) as refused:
            replayed.apply(logged[0].operation)
        assert refused.value.reason == "duplicate"


def test_apply_refuses_what_the_kernel_refuses_a_proposal(tmp_path: Path) -> None:
    """A host-supplied operation is not trusted because the host supplied it."""
    with _memory(tmp_path) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again, calmly")
        for operation, reason in (
            (
                MemoryOperation(intent=MemoryIntent.FORGET, target_ids=("missing",)),
                "unknown_target",
            ),
            # CORRECT acts on derived claims. An observation is evidence, not an inference.
            (
                MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(first.id,)),
                "not_derived",
            ),
            (
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(first.id,),
                    evidence_ids=(second.id,),
                ),
                "not_derived",
            ),
        ):
            with pytest.raises(ValidationError) as refused:
                memory.apply(operation)
            assert refused.value.reason == reason
        with pytest.raises(ValidationError):
            memory.apply(cast(Any, "forget everything"))

        # Cognitive forgetting through `apply` is the same operation `forget()` logs.
        record = memory.apply(MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id,)))
        assert record.forgotten_ids == (first.id,)
        assert memory.get(first.id).forgotten_at is not None
        assert memory.rollback(record.operation_id) is True
        assert memory.get(first.id).forgotten_at is None


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


def _consolidate_trait(
    memory: Memory,
    consolidator: ScriptedConsolidator,
    *sources: tuple[str | None, str],
) -> MemoryRecord:
    """Consolidate one trait out of one observation per `(place_id, household tag)` pair."""
    observed = _observations(
        memory,
        *(f"Ana waited calmly, time {index}" for index in range(len(sources))),
        places=[place_id for place_id, _household in sources],
        tags=[{"household": household} for _place_id, household in sources],
    )
    evidence_ids = tuple(record.id for record in observed)
    consolidator._scripts.append(
        (
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=evidence_ids,
                proposal=_trait("Ana", "patient"),
            ),
        )
    )
    report = memory.consolidate(evidence_ids=evidence_ids)
    assert report.rejected == ()
    return memory.get(report.operations[0].created_ids[0])


def test_a_consolidation_inherits_what_all_of_its_evidence_agrees_on(tmp_path: Path) -> None:
    """A derived record stands where its evidence stands -- when the evidence says one thing.

    Formation derives from one observation and inherits its place and metadata outright. A
    consolidation rests on several, so agreement is the condition: two sightings in the kitchen
    make kitchen knowledge, and a sighting in the kitchen plus one in the garage makes knowledge
    that belongs to neither room. `place_id` is a hard retrieval filter, so guessing would file
    the knowledge somewhere it was never observed.
    """
    agreeing = ScriptedConsolidator()
    with _memory(tmp_path / "agree", agreeing) as memory:
        derived = _consolidate_trait(memory, agreeing, ("kitchen", "flat-2"), ("kitchen", "flat-2"))

        assert derived.place_id == "kitchen"
        assert derived.metadata == {"household": "flat-2"}
        assert derived.id in {
            hit.id
            for hit in memory.search("patient", limit=10, scope=RetrievalScope(place_id="kitchen"))
        }

    disagreeing = ScriptedConsolidator()
    with _memory(tmp_path / "disagree", disagreeing) as memory:
        derived = _consolidate_trait(
            memory, disagreeing, ("kitchen", "flat-2"), ("garage", "flat-9")
        )

        assert derived.place_id is None
        assert derived.metadata == {}
        assert derived.id not in {
            hit.id
            for hit in memory.search("patient", limit=10, scope=RetrievalScope(place_id="kitchen"))
        }


def test_an_asserted_name_is_not_scoped_to_where_the_person_was_seen(tmp_path: Path) -> None:
    """Who somebody is does not stop being true in the next room.

    A naming assertion cites evidence but claims nothing about a place, so inheriting the place
    of the clips it was recognized in would hide the name from every question asked anywhere
    else -- and `register_identity`, which asserts the same thing with no evidence at all, files
    it nowhere.
    """
    consolidator = ScriptedConsolidator()
    with _identity_memory(tmp_path, consolidator) as memory:
        kitchen = ObservationContext(place_id="kitchen")
        first = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"), context=kitchen)
        second = memory.add(
            Blob(b"stranger speaks again", "video/mp4", "two.mp4"),
            context=kitchen,
            metadata={"household": "flat-2"},
        )
        identity_id = memory.faces(first.id)[0].identity_id
        for cited in (first.id, second.id):
            consolidator._scripts.append(
                (
                    MemoryOperation(
                        intent=MemoryIntent.IDENTIFY,
                        claim=IdentityClaim(identity_id=identity_id, name="Li"),
                        evidence_ids=(cited,),
                        rationale="the stranger said so",
                    ),
                )
            )
            assert not memory.consolidate(evidence_ids=(first.id, second.id)).rejected

        (asserted,) = [
            record
            for record in memory.list(limit=100).items
            if record.context is not None and record.context.identity_id == identity_id
        ]
        assert asserted.place_id is None
        assert asserted.metadata == {}
        assert asserted.id in {hit.id for hit in memory.search("recognized person", limit=10)}


# ---------------------------------------------------------------------------------------------
# The already-weighed marker


class SilentConsolidator:
    """Sees evidence and proposes nothing, which is the zero-yield pass the marker exists for."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self) -> None:
        self.calls = 0

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.calls += 1
        return ()

    def close(self) -> None:
        pass


def _trigger_rows(
    memory: Memory,
    trigger: MemoryTrigger,
    *,
    limit: int = 32,
    idle: bool = False,
) -> tuple[ConsolidationCandidate, ...]:
    rows = memory.consolidation_candidates(limit=limit, idle=idle)
    return tuple(row for row in rows if row.trigger is trigger)


def test_a_zero_yield_deliberation_stops_the_candidate_coming_back(tmp_path: Path) -> None:
    """A pass that proposed nothing still counts as weighed; a new signal makes it due again."""
    consolidator = SilentConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        first, _second = _observations(memory, "Ana waited calmly", "Ana waited again")
        memory.reinforce((first.id,))
        due = _trigger_rows(memory, MemoryTrigger.FEEDBACK)
        assert [row.memory_ids for row in due] == [(first.id,)]

        # The backend proposes nothing at all. Before the marker existed this pass left no trace
        # -- there is no operation row to derive consumption from -- so the same candidate came
        # back every round and was paid for every round.
        report = memory.consolidate(
            evidence_ids=due[0].memory_ids,
            trigger=MemoryTrigger.FEEDBACK,
        )
        assert report.operations == () and report.rejected == ()
        assert report.weighed == 1
        assert consolidator.calls == 1
        assert _trigger_rows(memory, MemoryTrigger.FEEDBACK) == ()

        # A confirmation after the attempt is a new signal, so it is due again.
        memory.reinforce((first.id,))
        assert [row.memory_ids for row in _trigger_rows(memory, MemoryTrigger.FEEDBACK)] == [
            (first.id,)
        ]


def test_a_contradiction_nothing_resolved_stops_being_relisted(tmp_path: Path) -> None:
    """A model that cannot settle a disagreement must not be asked about it every round."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        sources = _observations(
            memory,
            "Ana waited calmly",
            "Ana waited again, calmly",
            "Ana snapped at the delay",
            "Ana snapped again at the delay",
        )
        for pair, value in (((0, 1), "patient"), ((2, 3), "impatient")):
            cited = (sources[pair[0]].id, sources[pair[1]].id)
            consolidator._scripts.append(
                (
                    MemoryOperation(
                        intent=MemoryIntent.CONSOLIDATE,
                        evidence_ids=cited,
                        proposal=_trait("Ana", value),
                    ),
                )
            )
            memory.consolidate(evidence_ids=cited)

        due = _trigger_rows(memory, MemoryTrigger.CONTRADICTION)
        assert len(due) == 1
        # Weighed and unresolved: both claims still stand, and the lineage still disagrees, but
        # nothing about it has changed since the attempt.
        memory.consolidate(evidence_ids=due[0].memory_ids, trigger=MemoryTrigger.CONTRADICTION)
        assert _trigger_rows(memory, MemoryTrigger.CONTRADICTION) == ()


def test_repeated_recall_failure_becomes_a_candidate_naming_the_nearest_records(
    tmp_path: Path,
) -> None:
    """The QUERY_FAILURE producer: two near-equal empty recalls, one candidate."""
    with _memory(tmp_path, SilentConsolidator()) as memory:
        sources = _observations(memory, "the spare key is in the toolbox")
        # Fails because the store holds no procedural memory, not because the words are unknown.
        assert memory.search("where is the spare key", memory_type=MemoryType.PROCEDURAL) == ()
        assert _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE) == ()

        # Near-equal, not equal: case, punctuation, and spacing are normalized away.
        assert memory.search("Where  is the SPARE key?", memory_type=MemoryType.PROCEDURAL) == ()
        due = _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE)
        assert len(due) == 1
        assert due[0].evidence_count == 2
        # No evidence of its own -- that is what failing means -- so it names what the store does
        # hold nearest the question.
        assert sources[0].id in due[0].memory_ids

        memory.consolidate(evidence_ids=due[0].memory_ids, trigger=MemoryTrigger.QUERY_FAILURE)
        assert _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE) == ()


def test_a_query_failure_window_bounds_how_far_back_the_signal_counts(tmp_path: Path) -> None:
    with _memory(
        tmp_path,
        SilentConsolidator(),
        query_failure_window_seconds=0.001,
    ) as memory:
        _observations(memory, "the spare key is in the toolbox")
        for _attempt in range(2):
            assert memory.search("where is the key", memory_type=MemoryType.PROCEDURAL) == ()
        assert _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE) == ()


def test_memory_pressure_derives_candidates_only_over_a_declared_budget(tmp_path: Path) -> None:
    """PRESSURE is a configured budget, never growth on its own."""
    with _memory(tmp_path, SilentConsolidator()) as memory:
        _observations(memory, "one", "two", "three", "four")
        assert _trigger_rows(memory, MemoryTrigger.PRESSURE) == ()

    with _memory(tmp_path, SilentConsolidator(), memory_budget_records=2) as memory:
        due = _trigger_rows(memory, MemoryTrigger.PRESSURE)
        assert len(due) == 4
        assert {row.evidence_count for row in due} == {2}
        memory.consolidate(evidence_ids=due[0].memory_ids, trigger=MemoryTrigger.PRESSURE)
        assert len(_trigger_rows(memory, MemoryTrigger.PRESSURE)) == 3


def test_an_idle_window_is_declared_by_the_operator(tmp_path: Path) -> None:
    """IDLE is a parameter, not a clock the kernel reads."""
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        _observations(memory, "Ana waited calmly", "Ana waited again")
        assert _trigger_rows(memory, MemoryTrigger.IDLE) == ()

        due = _trigger_rows(memory, MemoryTrigger.IDLE, idle=True)
        assert due
        memory.consolidate(evidence_ids=due[0].memory_ids, trigger=MemoryTrigger.IDLE)
        weighed = set(due[0].memory_ids)
        assert all(
            not weighed & set(row.memory_ids)
            for row in _trigger_rows(memory, MemoryTrigger.IDLE, idle=True)
        )


# ---------------------------------------------------------------------------------------------
# The loop


class ResolvingConsolidator:
    """Retires the older side of any disagreement it is shown, so the loop reaches a fixed point."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self) -> None:
        self.calls = 0

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.calls += 1
        derived = [
            record
            for record in evidence
            if record.context is not None and record.context.kind is MemoryKind.TRAIT
        ]
        if trigger is not MemoryTrigger.CONTRADICTION or len(derived) < 2:
            return ()
        return (
            MemoryOperation(
                intent=MemoryIntent.CORRECT,
                target_ids=(derived[0].id,),
                rationale="the later reading stands",
            ),
        )

    def close(self) -> None:
        pass


def test_deliberate_runs_candidates_to_a_fixed_point(tmp_path: Path) -> None:
    """The loop entity: candidates -> consolidate -> repeat, ending because nothing is due."""
    scripted = ScriptedConsolidator()
    with _memory(tmp_path, scripted) as memory:
        sources = _observations(
            memory,
            "Ana waited calmly",
            "Ana waited again, calmly",
            "Ana snapped at the delay",
            "Ana snapped again at the delay",
        )
        for pair, value in (((0, 1), "patient"), ((2, 3), "impatient")):
            cited = (sources[pair[0]].id, sources[pair[1]].id)
            scripted._scripts.append(
                (
                    MemoryOperation(
                        intent=MemoryIntent.CONSOLIDATE,
                        evidence_ids=cited,
                        proposal=_trait("Ana", value),
                    ),
                )
            )
            memory.consolidate(evidence_ids=cited)

    resolver = ResolvingConsolidator()
    with _memory(tmp_path, resolver) as memory:
        assert _trigger_rows(memory, MemoryTrigger.CONTRADICTION)
        report = memory.deliberate(limit=8, max_rounds=5)
        assert isinstance(report, DeliberationReport)
        assert report.applied >= 1
        assert report.weighed >= 1
        assert report.model_calls == report.weighed
        # The ceiling was not what stopped it: nothing was due on the final round.
        assert report.rounds < 5
        assert memory.consolidation_candidates() == ()
        assert any(row.operation.intent is MemoryIntent.CORRECT for row in memory.operations())


def test_deliberate_does_not_loop_forever_on_a_backend_that_proposes_nothing(
    tmp_path: Path,
) -> None:
    """Zero yield must terminate and be reported as zero yield, not retried until the ceiling."""
    consolidator = SilentConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        sources = _observations(memory, "one", "two", "three")
        memory.reinforce(tuple(record.id for record in sources))

        report = memory.deliberate(limit=8, max_rounds=100)
        assert report.weighed == 3
        assert report.applied == 0
        assert report.rejected == 0
        assert report.rounds == 1
        assert consolidator.calls == 3
        assert memory.consolidation_candidates() == ()


def test_deliberate_needs_a_backend_and_validates_its_bounds(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        _observations(memory, "one")
        # Nothing is due in a store nobody has reinforced or consolidated, so the loop is a
        # no-op rather than a `ModelError`: `consolidate` is never reached.
        assert memory.deliberate() == DeliberationReport()
        for keywords in ({"limit": 0}, {"max_rounds": 0}, {"idle": "yes"}):
            with pytest.raises(ValidationError):
                memory.deliberate(**cast(Any, keywords))


# ---------------------------------------------------------------------------------------------
# Scheduling between latency-sensitive work and slow reasoning


class BlockingConsolidator:
    """Holds the pass open inside the backend round trip until the test releases it."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.entered.set()
        assert self.release.wait(timeout=30)
        return ()

    def close(self) -> None:
        pass


def test_a_thinking_backend_does_not_block_a_concurrent_add(tmp_path: Path) -> None:
    """`consolidate` releases the formation lock across the model round trip.

    Slow reasoning must not stall the latency-sensitive path. Correctness does not rest on the
    lock either: every proposal is re-checked inside its own apply transaction.
    """
    consolidator = BlockingConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        sources = _observations(memory, "Ana waited calmly", "Ana waited again")
        errors: list[BaseException] = []

        def deliberating() -> None:
            try:
                memory.consolidate(evidence_ids=tuple(record.id for record in sources))
            except BaseException as error:  # pragma: no cover - reported below
                errors.append(error)

        thread = threading.Thread(target=deliberating)
        thread.start()
        try:
            assert consolidator.entered.wait(timeout=30)
            # The backend is still thinking. Before the lock was split this blocked until it
            # returned.
            added = memory.add("Ana waited a third time", occurred_at=OCCURRED)
            assert memory.get(added.id).id == added.id
        finally:
            consolidator.release.set()
            thread.join(timeout=30)
        assert not thread.is_alive()
        assert errors == []


def test_an_identity_merge_does_not_commit_while_a_consolidate_apply_pass_holds_the_lock(
    tmp_path: Path,
) -> None:
    """The inverse of the test above: identity commits must serialize with formation too.

    A corroborated cross-modal MERGE is authorized and committed by the kernel itself, from
    `_link_asset_identity`, never through `consolidate()`'s own proposal pass -- so it is never
    covered by `_apply_memory_operation`'s formation-lock scope unless it takes the lock itself.
    Holding `_formation_lock` the way a consolidate apply pass does stands in for that pass here.
    """
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        transcriber=OnePersonSpeech(),
        face_analyzer=OnePersonFace(),
        identity_link_min_assets=1,
        minimum_relevance=0,
    ) as memory:
        record = memory.add(Blob(b"stranger arrives", "video/mp4", "one.mp4"))
        errors: list[BaseException] = []

        def merging() -> None:
            try:
                memory.faces(record.id)
            except BaseException as error:  # pragma: no cover - reported below
                errors.append(error)

        memory._formation_lock.acquire()
        try:
            thread = threading.Thread(target=merging)
            thread.start()
            thread.join(timeout=0.5)
            # The apply pass is "in progress" (simulated by holding its lock). Before this fix
            # the merge committed anyway, since `_link_asset_identity` took no lock of its own.
            assert thread.is_alive()
            assert memory.operations() == ()
        finally:
            memory._formation_lock.release()
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert errors == []
        assert [record.operation.intent for record in memory.operations()] == [MemoryIntent.MERGE]
        assert memory.faces(record.id)[0].identity_id == memory.speech(record.id)[0].speaker_id


# ---------------------------------------------------------------------------------------------
# Post-hoc outcome


def test_record_outcome_is_post_hoc_and_changes_nothing_else(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        first, second = _observations(memory, "Ana waited calmly", "Ana waited again, calmly")
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
        logged = report.operations[0]
        assert logged.outcome is None and logged.outcome_note is None

        assert memory.record_outcome(logged.operation_id, MemoryOutcome.CONFIRMED) is True
        judged = memory.operations()[0]
        assert judged.outcome is MemoryOutcome.CONFIRMED
        assert judged.outcome_note is None
        # Later evidence supersedes earlier evidence here as everywhere.
        assert (
            memory.record_outcome(
                logged.operation_id,
                MemoryOutcome.REFUTED,
                note="Ana was waiting for a delivery, not being patient",
            )
            is True
        )
        refuted = memory.operations()[0]
        assert refuted.outcome is MemoryOutcome.REFUTED
        assert refuted.outcome_note is not None
        # Purely a measurement: nothing was reversed and the derived record still stands.
        assert refuted.rolled_back_at is None
        assert memory.get(logged.created_ids[0]).id == logged.created_ids[0]

        assert memory.record_outcome(logged.operation_id + 999, MemoryOutcome.CONFIRMED) is False
        for arguments in (
            (0, MemoryOutcome.CONFIRMED),
            (logged.operation_id, "confirmed"),
        ):
            with pytest.raises(ValidationError):
                memory.record_outcome(*arguments)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            memory.record_outcome(logged.operation_id, MemoryOutcome.CONFIRMED, note=" ")


class Abstainer:
    """Refuses to answer from whatever it is shown, which is the abstention signal."""

    generation_capabilities = frozenset({Modality.TEXT})

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
        return AnswerResult(
            answer="I do not know",
            hits=(),
            abstained=True,
            abstention_reason=AbstentionReason.NO_EVIDENCE,
        )

    def close(self) -> None:
        pass


def test_an_abstention_and_a_thin_bundle_record_the_same_failure_signal(tmp_path: Path) -> None:
    """The other two QUERY_FAILURE producers: `ask` abstaining and `compile` finding nothing."""
    with _memory(tmp_path, SilentConsolidator(), answerer=Abstainer()) as memory:
        _observations(memory, "the spare key is in the toolbox")
        for _attempt in range(2):
            assert memory.ask("where is the passport").abstained is True
        due = _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE)
        assert len(due) == 1 and due[0].evidence_count == 2
        memory.consolidate(evidence_ids=due[0].memory_ids, trigger=MemoryTrigger.QUERY_FAILURE)
        assert _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE) == ()

        # A bundle with no evidence in it is the compiler reporting the same thing.
        for _attempt in range(2):
            assert (
                memory.compile(
                    "where is the passport",
                    budget=ContextBudget(memory_types=frozenset({MemoryType.PROCEDURAL})),
                ).hits
                == ()
            )
        assert _trigger_rows(memory, MemoryTrigger.QUERY_FAILURE)


def test_a_backend_may_not_propose_an_identity_merge(tmp_path: Path) -> None:
    """Merge authority is the kernel's, not a proposal vocabulary item.

    A cross-modal merge is committed from corroboration evidence the kernel counted itself, so
    a model that asks to fuse two people is refused before anything is read or written. An
    agent that can call ordinary recall must not be able to convert it into that authority.
    """
    consolidator = ScriptedConsolidator()
    with _memory(tmp_path / "merge", consolidator) as memory:
        first, second = _observations(memory, "Ana waited", "Ana waited again")
        proposed = MemoryOperation(
            intent=MemoryIntent.MERGE,
            identity=IdentityChange(identity_id="identity-1", moved_ids=("identity-2",)),
        )
        consolidator._scripts.append((proposed,))

        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert report.operations == ()
        assert report.rejected == ((proposed, "unauthorized"),)
        assert memory.operations() == ()
