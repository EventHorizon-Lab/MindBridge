"""A `RESPONSE_POLICY` is authorized by the host, never inferred by a model.

How the system should behave toward somebody is a grant. `docs/affective-memory.md` states the
rule; these tests are the executable half of it: the kernel refuses a model-proposed policy on
both derivation paths, and the host's own `apply()` writes one that carries
`RESPONSE_FEEDBACK`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    ObservationContext,
    ValidationError,
)

OCCURRED = datetime(2026, 3, 4, tzinfo=timezone.utc)


def _policy(subject: str = "Ana") -> FormationProposal:
    return FormationProposal(
        kind=MemoryKind.RESPONSE_POLICY,
        content=f"Answer {subject} briefly.",
        subject=subject,
        predicate="response_style",
        value="brief",
        confidence=0.9,
    )


class PolicyFormer:
    """Proposes one response policy and one ordinary entity from every observation."""

    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "policy-test"
    formation_space = "policy-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                _policy(),
                FormationProposal(
                    kind=MemoryKind.ENTITY,
                    content="Ana is a person",
                    subject="Ana",
                    confidence=0.9,
                ),
            )
            for _value in inputs
        )

    def close(self) -> None:
        pass


class PolicyConsolidator:
    """Proposes one response policy over whatever evidence it is shown."""

    consolidation_model = "policy-consolidator-test"
    consolidation_recipe = "policy-consolidator-test:v1"

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        del trigger
        return (
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=tuple(record.id for record in evidence),
                proposal=_policy(),
                rationale="she asked twice",
            ),
        )

    def close(self) -> None:
        pass


def _observations(memory: Memory, *contents: str) -> tuple[MemoryRecord, ...]:
    return tuple(
        memory.add(
            content,
            occurred_at=OCCURRED,
            context=ObservationContext(basis=EvidenceBasis.OBSERVATION, source_id=f"turn-{index}"),
        )
        for index, content in enumerate(contents)
    )


def _policies(memory: Memory) -> list[MemoryRecord]:
    return [
        record
        for record in memory.list(limit=100).items
        if record.context is not None and record.context.kind is MemoryKind.RESPONSE_POLICY
    ]


def test_a_response_policy_a_former_proposes_is_refused(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PolicyFormer(),
        minimum_relevance=0,
    ) as memory:
        _observations(memory, "Ana asked for shorter answers")

        assert _policies(memory) == []
        assert memory.search("briefly", limit=10, memory_type=MemoryType.PROCEDURAL) == ()
        # The refusal costs that one proposal: its sibling from the same observation stands.
        assert any(
            record.context is not None and record.context.kind is MemoryKind.ENTITY
            for record in memory.list(limit=100).items
        )


def test_a_response_policy_a_consolidator_proposes_is_refused(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=PolicyConsolidator(),
        minimum_relevance=0,
    ) as memory:
        first, second = _observations(memory, "Ana asked for less", "Ana asked for less again")

        report = memory.consolidate(evidence_ids=(first.id, second.id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["invalid_proposal"]
        assert _policies(memory) == []


def test_a_host_authored_response_policy_is_stored_as_response_feedback(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=PolicyConsolidator(),
        minimum_relevance=0,
    ) as memory:
        first, second = _observations(memory, "Ana asked for less", "Ana asked for less again")

        record = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(first.id, second.id),
                proposal=_policy(),
                rationale="the household owner asked for it",
            )
        )

        derived = memory.get(record.created_ids[0])
        assert derived.context is not None
        assert derived.context.kind is MemoryKind.RESPONSE_POLICY
        # The authorization the record names, not the `MODEL_INFERENCE` default it was built with.
        assert derived.context.basis is EvidenceBasis.RESPONSE_FEEDBACK
        assert derived.context.visible is True
        assert derived.memory_type is MemoryType.PROCEDURAL
        assert [
            hit.id for hit in memory.search("briefly", limit=10, memory_type=MemoryType.PROCEDURAL)
        ] == [derived.id]
        assert [procedure.id for procedure in memory.compile("briefly").procedures] == [derived.id]


def test_a_logged_response_policy_replays_as_the_duplicate_it_is(tmp_path: Path) -> None:
    """The authorization is stamped on the stored record, so the logged proposal stays replayable.

    Pinning the basis on the proposal itself re-keyed it: the derived ID is computed from the
    proposal's basis, so a logged row -- and any row written before the rule existed, carrying
    `model_inference` in its JSON -- minted a second ID on replay instead of being recognised
    as already applied.
    """
    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        first, second = _observations(memory, "Ana asked for less", "Ana asked for less again")
        operation = MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(first.id, second.id),
            proposal=_policy(),
            rationale="the household owner asked for it",
        )

        applied = memory.apply(operation)
        logged = memory.operations()[0]
        assert logged.operation_id == applied.operation_id
        assert logged.operation.proposal is not None
        # The log keeps the proposal it was handed, defaults and all.
        assert logged.operation.proposal.basis is EvidenceBasis.MODEL_INFERENCE

        for replayed in (logged.operation, operation):
            with pytest.raises(ValidationError) as refused:
                memory.apply(replayed)
            assert refused.value.reason == "duplicate"

        policies = _policies(memory)
        assert len(policies) == 1
        stored = policies[0].context
        assert stored is not None
        assert stored.basis is EvidenceBasis.RESPONSE_FEEDBACK


def test_a_response_policy_proposal_still_cannot_claim_an_observation() -> None:
    """The generic rule stands: nothing formed may claim to have been observed directly."""
    with pytest.raises(ValidationError):
        FormationProposal(
            kind=MemoryKind.RESPONSE_POLICY,
            content="Answer Ana briefly.",
            basis=EvidenceBasis.OBSERVATION,
            subject="Ana",
            predicate="response_style",
            value="brief",
            confidence=0.9,
        )
