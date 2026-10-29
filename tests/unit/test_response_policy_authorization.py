"""A `RESPONSE_POLICY` is authorized by the host, never inferred by a model.

How the system should behave toward somebody is a grant. `docs/affective-memory.md` states the
rule; these tests are the executable half of it: the kernel refuses a model-proposed policy on
both derivation paths, and the host's own `apply()` writes one that carries
`RESPONSE_FEEDBACK`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from test_memory_control_plane import OnePersonFace, OnePersonSpeech

from mindbridge import (
    Blob,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    IdentityClaim,
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


class OperationConsolidator:
    consolidation_model = "operation-test"
    consolidation_recipe = "operation-test:v1"

    def __init__(self) -> None:
        self.proposals: tuple[MemoryOperation, ...] = ()
        self.before_propose: Callable[[], None] | None = None

    def consolidate(
        self, evidence: Sequence[MemoryRecord], *, trigger: MemoryTrigger
    ) -> tuple[MemoryOperation, ...]:
        if self.before_propose is not None:
            self.before_propose()
        return self.proposals

    def close(self) -> None:
        pass


def _apply_policy(memory: Memory, source_id: str) -> MemoryRecord:
    operation = memory.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(source_id,),
            proposal=_policy(),
        )
    )
    return memory.get(operation.created_ids[0])


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(
    "intent",
    [MemoryIntent.FORGET, MemoryIntent.CORRECT, MemoryIntent.CONSOLIDATE, MemoryIntent.REINFORCE],
)
def test_model_cannot_change_an_authorized_policy(
    tmp_path: Path, independent: bool, intent: MemoryIntent
) -> None:
    backend = OperationConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=backend,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        source, other = _observations(memory, "Ana asked for brief answers", "Ana repeats it")
        policy = _apply_policy(memory, source.id)
        backend.proposals = (
            MemoryOperation(
                intent=intent,
                target_ids=(policy.id,),
                evidence_ids=(
                    (other.id,)
                    if intent is MemoryIntent.REINFORCE
                    else (policy.id,)
                    if intent is MemoryIntent.CONSOLIDATE
                    else ()
                ),
                proposal=(
                    FormationProposal(
                        kind=MemoryKind.EVENT, content="Ana asked for less.", confidence=0.9
                    )
                    if intent is MemoryIntent.CONSOLIDATE
                    else None
                ),
            ),
        )

        report = memory.consolidate(evidence_ids=(policy.id, other.id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["unauthorized"]
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("target", ["observation", "derived"])
def test_model_cannot_forget_a_policy_ancestor_and_preserves_unrelated_work(
    tmp_path: Path, independent: bool, target: str
) -> None:
    backend = OperationConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=backend,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        source, unrelated = _observations(memory, "Ana asked for brief answers", "Unrelated note")
        middle = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=MemoryKind.EVENT, content="Ana requested concise replies.", confidence=0.9
                ),
            )
        )
        policy = _apply_policy(memory, middle.created_ids[0])
        target_id = source.id if target == "observation" else middle.created_ids[0]
        backend.proposals = (
            MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(target_id, unrelated.id)),
            MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(unrelated.id,)),
        )

        report = memory.consolidate(evidence_ids=(target_id, unrelated.id))

        assert [reason for _operation, reason in report.rejected] == ["unauthorized"]
        assert len(report.operations) == 1
        assert memory.get(target_id).forgotten_at is None
        assert memory.get(unrelated.id).forgotten_at is not None
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}
        # The host retains authority to withdraw and restore the same supporting observation.
        forgotten = memory.forget((target_id,))
        assert forgotten is not None
        assert memory.rollback(forgotten.operation_id)
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_model_cannot_implicitly_supersede_a_policy_ancestor(
    tmp_path: Path, independent: bool, partial: bool
) -> None:
    backend = OperationConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=backend,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        source, new = _observations(memory, "Ana lives in Paris", "Ana moved to Rome")
        ancestor = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=MemoryKind.STATE,
                    content="Ana lives in Paris.",
                    subject="Ana",
                    predicate="home",
                    value="Paris",
                    confidence=0.8,
                    valid_from=OCCURRED,
                ),
            )
        )
        policy = _apply_policy(memory, ancestor.created_ids[0])
        replacement = MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(new.id,),
            proposal=FormationProposal(
                kind=MemoryKind.STATE,
                content="Ana lives in Rome.",
                subject="Ana",
                predicate="home",
                value="Rome",
                confidence=0.9,
                valid_from=datetime(2026, 3, 5, tzinfo=timezone.utc) if partial else OCCURRED,
                valid_until=datetime(2026, 3, 6, tzinfo=timezone.utc) if partial else None,
            ),
        )
        backend.proposals = (replacement,)
        before = memory.get(ancestor.created_ids[0]).context

        report = memory.consolidate(evidence_ids=(new.id,))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["unauthorized"]
        assert memory.get(ancestor.created_ids[0]).context == before
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}
        # An explicit host correction can replace the same state.
        assert memory.apply(replacement).created_ids


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(
    "intent", [MemoryIntent.CORRECT, MemoryIntent.REINFORCE, MemoryIntent.CONSOLIDATE]
)
def test_model_cannot_mutate_policy_support_or_replace_it_during_consolidation(
    tmp_path: Path, independent: bool, intent: MemoryIntent
) -> None:
    backend = OperationConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=backend,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        source, other = _observations(memory, "Ana requested brief replies", "A separate request")
        ancestor = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=MemoryKind.EVENT, content="Ana requested brevity.", confidence=0.9
                ),
            )
        )
        ancestor_id = ancestor.created_ids[0]
        policy = _apply_policy(memory, ancestor_id)
        before = memory.get(ancestor_id)
        backend.proposals = (
            MemoryOperation(
                intent=intent,
                target_ids=(ancestor_id,),
                evidence_ids=(
                    (other.id,)
                    if intent is MemoryIntent.REINFORCE
                    else (ancestor_id,)
                    if intent is MemoryIntent.CONSOLIDATE
                    else ()
                ),
                proposal=(
                    FormationProposal(
                        kind=MemoryKind.EVENT, content="A replacement summary.", confidence=0.9
                    )
                    if intent is MemoryIntent.CONSOLIDATE
                    else None
                ),
            ),
        )

        report = memory.consolidate(evidence_ids=(ancestor_id, other.id))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["unauthorized"]
        assert memory.get(ancestor_id) == before
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}


class ReplacingFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "replacement-test"
    formation_space = "replacement-test:v1"

    def __init__(self) -> None:
        self.proposals: tuple[FormationProposal, ...] = (
            FormationProposal(
                kind=MemoryKind.STATE,
                content="Ana lives in Paris.",
                subject="Ana",
                predicate="home",
                value="Paris",
                confidence=0.8,
            ),
        )

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(self.proposals for _value in inputs)

    def close(self) -> None:
        pass


class BatchReplacingFormer(ReplacingFormer):
    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                *self.proposals,
                FormationProposal(
                    kind=MemoryKind.STATE,
                    content=f"Ben lives in {value.content.text}.",
                    subject="Ben",
                    predicate="home",
                    value=value.content.text,
                    confidence=0.9,
                ),
            )
            for value in inputs
        )


@pytest.mark.parametrize("independent", [False, True])
def test_refusing_policy_replacement_preserves_independent_batch_state_conflicts(
    tmp_path: Path, independent: bool
) -> None:
    former = BatchReplacingFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        memory.add("Madrid", occurred_at=OCCURRED)
        ancestor = next(
            record
            for record in memory.list().items
            if record.context is not None and record.context.subject == "Ana"
        )
        policy = _apply_policy(memory, ancestor.id)
        former.proposals = (
            FormationProposal(
                kind=MemoryKind.STATE,
                content="Ana lives in Rome.",
                subject="Ana",
                predicate="home",
                value="Rome",
                confidence=0.9,
            ),
        )

        memory.add_many(("Paris", "London"), occurred_at=(OCCURRED, OCCURRED))

        siblings = [
            record.context
            for record in memory.list().items
            if record.context is not None
            and record.context.subject == "Ben"
            and record.context.value in {"Paris", "London"}
        ]
        assert len(siblings) == 2
        assert all(context.retired_at is None for context in siblings)
        assert memory.get(ancestor.id).context == ancestor.context
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}


def test_host_can_reinforce_a_naming_assertion_supporting_a_response_policy(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        transcriber=OnePersonSpeech(),
        face_analyzer=OnePersonFace(),
        independent_evidence=True,
        identity_link_min_assets=1,
        minimum_relevance=0,
    ) as memory:
        sources = tuple(
            memory.add(
                Blob(f"Ana visit {index}".encode(), "video/mp4", f"visit-{index}.mp4"),
                context=ObservationContext(source_id=f"visit-{index}"),
            )
            for index in range(3)
        )
        identity_id = memory.faces(sources[0].id)[0].identity_id
        claim = IdentityClaim(identity_id=identity_id, name="Ana")
        naming_operations = [
            memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.IDENTIFY, claim=claim, evidence_ids=(source.id,)
                )
            )
            for source in sources[:2]
        ]
        naming_id = naming_operations[0].created_ids[0]
        policy = _apply_policy(memory, naming_id)

        applied = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.IDENTIFY, claim=claim, evidence_ids=(sources[2].id,)
            )
        )

        assert applied.operation.intent is MemoryIntent.IDENTIFY
        naming_context = memory.get(naming_id).context
        assert naming_context is not None
        assert set(naming_context.evidence_ids) == {source.id for source in sources}
        policy_context = memory.get(policy.id).context
        assert policy_context is not None
        assert policy_context.kind is MemoryKind.RESPONSE_POLICY
        assert policy_context.basis is EvidenceBasis.RESPONSE_FEEDBACK
        assert policy_context.retired_at is None


@pytest.mark.parametrize("independent", [False, True])
def test_automatic_formation_refuses_policy_replacement_and_keeps_valid_siblings(
    tmp_path: Path, independent: bool
) -> None:
    former = ReplacingFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        memory.add("Ana lives in Paris", occurred_at=OCCURRED)
        ancestor = next(
            record
            for record in memory.list().items
            if record.context is not None and record.context.kind is MemoryKind.STATE
        )
        policy = _apply_policy(memory, ancestor.id)
        former.proposals = (
            FormationProposal(
                kind=MemoryKind.STATE,
                content="Ana lives in Rome.",
                subject="Ana",
                predicate="home",
                value="Rome",
                confidence=0.9,
            ),
            FormationProposal(
                kind=MemoryKind.EVENT, content="A separate supported event.", confidence=0.9
            ),
        )

        added = memory.add("Ana moved to Rome and attended a meeting", occurred_at=OCCURRED)

        assert memory.get(added.id).content == added.content
        assert memory.get(ancestor.id).context == ancestor.context
        assert policy.id in {record.id for record in memory.compile("briefly").procedures}
        assert any(
            record.content == "A separate supported event." for record in memory.list().items
        )


@pytest.mark.parametrize("independent", [False, True])
def test_policy_created_during_model_work_is_protected_by_the_apply_transaction(
    tmp_path: Path, independent: bool
) -> None:
    backend = OperationConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=backend,
        minimum_relevance=0,
        independent_evidence=independent,
    ) as memory:
        (source,) = _observations(memory, "Ana requested brief replies")

        def create_policy() -> None:
            _apply_policy(memory, source.id)

        backend.before_propose = create_policy
        backend.proposals = (MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(source.id,)),)
        report = memory.consolidate(evidence_ids=(source.id,))

        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["unauthorized"]
        assert memory.compile("briefly").procedures


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
