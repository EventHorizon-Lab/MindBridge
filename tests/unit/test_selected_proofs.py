"""Delivery certificates preserve provenance, corroboration and actual budget costs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_independent_evidence import _derive, _open, _root

from mindbridge import ContextBudget, ContextProof, ContextProofNode, ValidationError
from mindbridge.api.app import ContextBundleResponse
from mindbridge.api.content import ContextBudgetInput, context_budget
from mindbridge.api.mcp import ContextBundleResult
from mindbridge.kernel.corroboration import (
    Assessment,
    EvidenceNode,
    SupportSummary,
    independent_support,
)
from mindbridge.kernel.proofs import certificates, proof_support, select_certificates


def test_count_and_confidence_can_require_three_separate_witnesses() -> None:
    nodes = {
        "a": EvidenceNode("capture-a"),
        "b": EvidenceNode("capture-b"),
        "target": EvidenceNode(
            None,
            assessments=(
                Assessment(("a",), 0.2),
                Assessment(("b",), 0.2),
                Assessment(("a", "b"), 0.99),
            ),
        ),
    }
    proofs, truncated = certificates(nodes, "target")
    required = independent_support(nodes, "target")
    bought = select_certificates(proofs, required, dict.fromkeys(nodes, 1))
    assert not truncated
    assert len(bought) == 3
    assert proof_support(bought) == (2, 0.99)
    assert {step.memory_id for proof in bought for step in proof.nodes} == set(nodes)


@pytest.mark.parametrize("steps", range(1, 14))
def test_work_exhaustion_cannot_certify_an_unfinished_and(steps: int) -> None:
    nodes = {
        "a": EvidenceNode("shared"),
        "b": EvidenceNode("shared"),
        "target": EvidenceNode(None, assessments=(Assessment(("a", "b"), 0.7),)),
    }
    proofs, _ = certificates(nodes, "target", max_steps=steps)
    for proof in proofs:
        assert {step.memory_id for step in proof.nodes} == set(nodes)
    count, confidence = proof_support(proofs)
    assert count <= 1 and confidence <= 0.7


def test_cycles_and_missing_sources_never_produce_certificates() -> None:
    nodes = {
        "a": EvidenceNode(None, assessments=(Assessment(("b",), 1.0),)),
        "b": EvidenceNode(
            None, assessments=(Assessment(("a",), 1.0), Assessment(("missing",), 1.0))
        ),
    }
    assert certificates(nodes, "a")[0] == ()


def test_repeated_capture_cannot_satisfy_two_witness_obligation() -> None:
    nodes = {
        "a": EvidenceNode("shared"),
        "b": EvidenceNode("shared"),
        "target": EvidenceNode(
            None, assessments=(Assessment(("a",), 0.6), Assessment(("b",), 0.6))
        ),
    }
    proofs, _ = certificates(nodes, "target")
    assert select_certificates(proofs, SupportSummary(2, 0.6, False), dict.fromkeys(nodes, 1)) == ()


@pytest.mark.parametrize("sources", [("missing",), ("target",)])
def test_public_proof_rejects_incomplete_or_cyclic_certificate(sources: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        ContextProof(
            anchor_id="target",
            nodes=(ContextProofNode(memory_id="target", sources=sources, confidence=0.6),),
        )


def test_sdk_delivers_certificates_without_rewriting_the_audit_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _open(tmp_path) as memory:
        roots = tuple(_root(memory, f"evidence {index}") for index in range(8))
        target = _derive(memory, (roots[0],), "patient", trait=True).created_ids[0]
        for root in roots[1:]:
            _derive(memory, (root,), "patient", trait=True)
        stored = memory._store.records.read_memory(target)
        assert stored is not None and stored.context is not None
        audit_ids = stored.context.evidence_ids
        monkeypatch.setattr(
            memory._retrieval,
            "search_prepared",
            lambda *args, **kwargs: SimpleNamespace(
                hits=(memory._hydrator.search_hit(stored, 0.9),), matched_dense_index_ids={}
            ),
        )
        full = memory.compile("patient", budget=ContextBudget(max_items=3))
        selected = memory.compile(
            "patient", budget=ContextBudget(max_items=3, selected_proofs=True)
        )
        assert not full.hits
        assert len(selected.hits) == 3
        assert {hit.id for hit in selected.hits} >= {target}
        delivered = next(hit for hit in selected.hits if hit.id == target)
        assert delivered.context is not None and delivered.context.evidence_ids == audit_ids
        assert memory.get(target).context == stored.context
        assert proof_support(
            tuple(proof for proof in selected.proofs if proof.anchor_id == target)
        ) == pytest.approx((2, 0.84))
        # The fixed header reserves the maximum digit width, as in ordinary compilation.
        assert 0 <= selected.chars - len(selected.render().split("\n\n## Unknowns")[0]) <= 4
        assert selected.chars <= selected.budget.max_chars
        assert "## Selected proofs" in selected.compact().text
        assert target not in selected.compact().text
        for model in (ContextBundleResponse, ContextBundleResult):
            wire = model.model_validate(selected.document()).model_dump(mode="json")
            assert wire["budget"]["selected_proofs"] is True
            assert wire["proofs"][0]["anchor_id"] == target
        too_small = memory.compile(
            "patient",
            budget=ContextBudget(
                max_items=3,
                max_chars=selected.chars - 5,
                selected_proofs=True,
            ),
        )
        assert not too_small.hits
        with pytest.raises(ValidationError, match="confidence"):
            replace(selected, proofs=selected.proofs[:1])
        with pytest.raises(ValidationError, match="full hit"):
            replace(selected, facts=())


def test_budget_input_preserves_opt_in_and_rejects_non_boolean() -> None:
    budget = context_budget(ContextBudgetInput(selected_proofs=True))
    assert budget is not None and budget.selected_proofs
    with pytest.raises(ValidationError):
        ContextBudget(selected_proofs=1)  # type: ignore[arg-type]
