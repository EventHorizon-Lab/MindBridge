"""Public SDK checks for the opt-in independent proof projection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AsyncMemory,
    ContextBudget,
    EvidenceBasis,
    FormationProposal,
    Memory,
    MemoryConfig,
    MemoryIntent,
    MemoryKind,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryPlugins,
    ObservationContext,
    StorageError,
    ValidationError,
)
from mindbridge.infrastructure.local import LocalStore


def _open(path: Path, independent: bool = True) -> Memory:
    return Memory.from_plugins(
        path,
        plugins=MemoryPlugins(embedder=TinyEmbedder()),
        config=MemoryConfig(independent_evidence=independent, minimum_relevance=0),
    )


def _root(memory: Memory, name: str, capture: str | None = None) -> str:
    return memory.add(
        name,
        context=ObservationContext(basis=EvidenceBasis.OBSERVATION, source_id=capture),
    ).id


def _derive(
    memory: Memory,
    sources: tuple[str, ...],
    name: str,
    *,
    trait: bool = False,
    basis: EvidenceBasis = EvidenceBasis.MODEL_INFERENCE,
    forget: bool = False,
) -> MemoryOperationRecord:
    return memory.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=sources,
            target_ids=sources if forget else (),
            proposal=FormationProposal(
                kind=MemoryKind.TRAIT if trait else MemoryKind.RELATION,
                basis=basis,
                content=name,
                subject="Ana",
                predicate="disposition" if trait else "event",
                value=name,
                confidence=0.6,
            ),
        )
    )


def _visible(memory: Memory, memory_id: str) -> bool:
    context = memory.get(memory_id).context
    assert context is not None
    return context.visible


@pytest.mark.parametrize("independent", [False, True])
def test_two_disjoint_joint_assessments_and_withdrawal(tmp_path: Path, independent: bool) -> None:
    with _open(tmp_path, independent) as memory:
        roots = tuple(_root(memory, f"event {index}") for index in range(4))
        first = _derive(memory, roots[:2], "patient", trait=True)
        target = first.created_ids[0]
        assert not _visible(memory, target)
        second = _derive(memory, roots[2:], "patient", trait=True)
        assert second.changed_ids == (target,)
        assert _visible(memory, target) is independent
        context = memory.get(target).context
        assert context is not None
        assert context.confidence == pytest.approx(0.84 if independent else 0.6)
        assert (target in {hit.id for hit in memory.search("patient", limit=20)}) is independent
        bundle = memory.compile("patient", budget=ContextBudget(max_chars=20_000))
        assert (target in {hit.id for hit in bundle.hits}) is independent
        assert memory.rollback(second.operation_id)
        assert not _visible(memory, target)
        _derive(memory, roots[2:], "patient", trait=True)
        assert memory.delete(roots[2])
        assert not _visible(memory, target)
    with _open(tmp_path, independent) as reopened:
        assert not _visible(reopened, target)


@pytest.mark.parametrize("shared_capture", [False, True])
def test_repeated_capture_and_shared_multiroot_summaries_stay_hidden(
    tmp_path: Path, shared_capture: bool
) -> None:
    with _open(tmp_path) as memory:
        a = _root(memory, "first event", "shared" if shared_capture else None)
        b = _root(memory, "second event", "shared" if shared_capture else None)
        c = _root(memory, "third event")
        left = _derive(memory, (a, b), "left summary").created_ids[0]
        right_sources = (c, b) if not shared_capture else (b,)
        right = _derive(memory, right_sources, "right summary").created_ids[0]
        target = _derive(memory, (left,), "patient", trait=True).created_ids[0]
        _derive(memory, (right,), "patient", trait=True)
        assert not _visible(memory, target)


def test_or_reinforcement_preserves_existing_independent_proof(tmp_path: Path) -> None:
    with _open(tmp_path) as memory:
        a, b, c = tuple(_root(memory, name) for name in ("a", "b", "c"))
        left = _derive(memory, (a,), "left summary").created_ids[0]
        right = _derive(memory, (b,), "right summary").created_ids[0]
        target = _derive(memory, (left,), "patient", trait=True).created_ids[0]
        _derive(memory, (right,), "patient", trait=True)
        assert _visible(memory, target)
        _derive(memory, (b, c), "left summary")
        assert _visible(memory, target)


def test_nested_provenance_refreshes_without_scalar_group_change(tmp_path: Path) -> None:
    with _open(tmp_path) as memory:
        a, b, c, d = tuple(_root(memory, name) for name in ("a", "b", "c", "d"))
        left = _derive(memory, (a, b), "left summary").created_ids[0]
        nested = _derive(memory, (left,), "nested summary").created_ids[0]
        right = _derive(memory, (b, c), "right summary").created_ids[0]
        target = _derive(memory, (nested,), "patient", trait=True).created_ids[0]
        _derive(memory, (right,), "patient", trait=True)
        assert not _visible(memory, target)
        change = _derive(memory, (a, d), "left summary")
        assert _visible(memory, target)
        assert memory.rollback(change.operation_id)
        assert not _visible(memory, target)


@pytest.mark.parametrize("independent", [False, True])
def test_projection_policy_is_pinned_across_reopen(tmp_path: Path, independent: bool) -> None:
    with _open(tmp_path, independent):
        pass
    with pytest.raises(StorageError, match=r"evidence\.projection_recipe"):
        _open(tmp_path, not independent)
    with _open(tmp_path, independent):
        pass


def test_unmarked_existing_legacy_store_cannot_switch_policy(tmp_path: Path) -> None:
    with _open(tmp_path, False) as memory:
        _root(memory, "legacy record")
    with LocalStore(tmp_path) as store:
        store.delete_metadata("evidence.projection_recipe")
    with pytest.raises(StorageError, match=r"evidence\.projection_recipe"):
        _open(tmp_path, True)
    with _open(tmp_path, False) as memory:
        assert memory.list().items


def test_unmarked_empty_store_can_select_candidate(tmp_path: Path) -> None:
    with LocalStore(tmp_path):
        pass
    with _open(tmp_path) as memory:
        assert memory.list().items == ()


@pytest.mark.parametrize("independent", [False, True])
def test_reopen_preserves_live_joint_support(tmp_path: Path, independent: bool) -> None:
    with _open(tmp_path, independent) as memory:
        roots = tuple(_root(memory, f"event {index}") for index in range(4))
        target = _derive(memory, roots[:2], "patient", trait=True).created_ids[0]
        _derive(memory, roots[2:], "patient", trait=True)
    with _open(tmp_path, independent) as memory:
        assert _visible(memory, target) is independent
        assert (target in {hit.id for hit in memory.search("patient", limit=20)}) is independent


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_policy_requires_a_strict_boolean(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(ValidationError, match="independent_evidence"):
        _open(tmp_path, cast(bool, invalid))
    assert not (tmp_path / "state.sqlite3").exists()


def test_default_and_async_configuration(tmp_path: Path) -> None:
    assert MemoryConfig().independent_evidence is False

    async def run() -> None:
        async with AsyncMemory.from_plugins(
            tmp_path,
            plugins=MemoryPlugins(embedder=TinyEmbedder()),
            config=MemoryConfig(independent_evidence=True),
        ) as memory:
            await memory.add("async observation")

    asyncio.run(run())
    with pytest.raises(StorageError, match=r"evidence\.projection_recipe"):
        _open(tmp_path, False)


def test_capture_label_can_alias_an_unlabelled_record_id(tmp_path: Path) -> None:
    with _open(tmp_path) as memory:
        first = _root(memory, "unlabelled capture")
        second = _root(memory, "explicitly labelled capture", first)
        target = _derive(memory, (first,), "patient", trait=True).created_ids[0]
        _derive(memory, (second,), "patient", trait=True)
        assert not _visible(memory, target)


@pytest.mark.parametrize("limit_name", ["_MAX_NODES", "_MAX_MEMBER_ROWS"])
def test_truncated_ancestry_does_not_fabricate_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    limit_name: str,
) -> None:
    from mindbridge.infrastructure.local.store import _corroboration

    monkeypatch.setattr(_corroboration, limit_name, 3)
    with _open(tmp_path) as memory:
        roots = tuple(_root(memory, f"event {index}") for index in range(4))
        target = _derive(memory, roots[:2], "patient", trait=True).created_ids[0]
        _derive(memory, roots[2:], "patient", trait=True)
        assert not _visible(memory, target)
        assert "support lower bound" in caplog.text


def test_root_confidence_constrains_nested_assessments(tmp_path: Path) -> None:
    with _open(tmp_path) as memory:
        root = memory.add(
            "uncertain capture",
            context=ObservationContext(basis=EvidenceBasis.OBSERVATION, confidence=0.2),
        ).id
        summary = _derive(memory, (root,), "summary").created_ids[0]
        target = _derive(memory, (summary,), "patient", trait=True).created_ids[0]
        context = memory.get(target).context
        assert context is not None
        assert context.confidence == pytest.approx(0.2)


@pytest.mark.parametrize("basis", [EvidenceBasis.USER_STATEMENT, EvidenceBasis.RESPONSE_FEEDBACK])
def test_derived_host_basis_cannot_erase_ancestry(tmp_path: Path, basis: EvidenceBasis) -> None:
    with _open(tmp_path) as memory:
        root = _root(memory, "one capture")
        left = _derive(memory, (root,), "left", basis=basis).created_ids[0]
        right = _derive(memory, (root,), "right", basis=basis).created_ids[0]
        target = _derive(memory, (left,), "patient", trait=True).created_ids[0]
        _derive(memory, (right,), "patient", trait=True)
        assert not _visible(memory, target)
        assert memory.delete(root)
        for memory_id in (left, right, target):
            with pytest.raises(MemoryNotFoundError):
                memory.get(memory_id)


def test_consolidation_forgetting_preserves_ancestral_support(tmp_path: Path) -> None:
    with _open(tmp_path) as memory:
        first, second = (_root(memory, name) for name in ("first capture", "second capture"))
        summary = _derive(memory, (first,), "summary", forget=True).created_ids[0]
        target = _derive(memory, (summary,), "patient", trait=True).created_ids[0]
        _derive(memory, (second,), "patient", trait=True)
        assert _visible(memory, target)
        assert memory.get(first).forgotten_at is not None
        assert memory.delete(first)
        assert not _visible(memory, target)
