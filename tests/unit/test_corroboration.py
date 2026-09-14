"""Capture-grounded support must survive alternatives without laundering sources."""

from dataclasses import replace

import pytest

from mindbridge.kernel.corroboration import (
    Assessment,
    EvidenceNode,
    SupportSummary,
    independent_support,
)


def derived(*sources: tuple[str, ...], confidence: float = 0.8) -> EvidenceNode:
    return EvidenceNode(None, assessments=tuple(Assessment(s, confidence) for s in sources))


@pytest.mark.parametrize("same_capture, count, confidence", [(True, 1, 0.8), (False, 2, 0.96)])
def test_capture_identity_controls_independence(
    same_capture: bool, count: int, confidence: float
) -> None:
    nodes = {
        "a": EvidenceNode("capture-a"),
        "b": EvidenceNode("capture-a" if same_capture else "capture-b"),
        "target": derived(("a",), ("b",)),
    }
    result = independent_support(nodes, "target")
    assert result.count == count
    assert result.confidence == pytest.approx(confidence)
    assert not result.truncated


def test_disjoint_joint_assessments_corroborate() -> None:
    nodes = {name: EvidenceNode(name) for name in "abcd"}
    nodes["target"] = derived(("a", "b"), ("c", "d"))
    result = independent_support(nodes, "target")
    assert result.count == 2
    assert result.confidence == pytest.approx(0.96)


def test_shared_intermediate_cannot_choose_different_roots_to_corroborate() -> None:
    nodes = {name: EvidenceNode(name) for name in "ab"}
    nodes.update(
        shared=derived(("a",), ("b",)),
        left=derived(("shared",)),
        right=derived(("shared",)),
        target=derived(("left",), ("right",)),
    )
    assert independent_support(nodes, "target") == SupportSummary(1, 0.8, False)


def test_or_alternatives_preserve_old_independent_proofs() -> None:
    nodes = {name: EvidenceNode(name) for name in "ab"}
    nodes.update(summary=derived(("a",)), target=derived(("summary",), ("b",)))
    before = independent_support(nodes, "target")
    nodes["summary"] = derived(("a",), ("b",))
    assert independent_support(nodes, "target") == before
    assert before.count == 2


def test_alternatives_of_one_top_assessment_are_only_one_witness() -> None:
    nodes = {name: EvidenceNode(name) for name in "ab"}
    nodes.update(summary=derived(("a",), ("b",)), target=derived(("summary",)))
    assert independent_support(nodes, "target") == SupportSummary(1, 0.8, False)


def test_joint_clause_combines_complete_alternative_proofs() -> None:
    nodes = {name: EvidenceNode(name) for name in "abcd"}
    nodes.update(
        left=derived(("a",), ("b",)),
        right=derived(("c",), ("d",)),
        target=derived(("left", "right"), ("a", "c")),
    )
    result = independent_support(nodes, "target")
    assert result.count == 2
    assert result.confidence == pytest.approx(0.96)


def test_support_count_and_confidence_can_have_different_witnesses() -> None:
    nodes = {name: EvidenceNode(name) for name in "ab"}
    nodes["target"] = EvidenceNode(
        None,
        assessments=(
            Assessment(("a", "b"), 0.99),
            Assessment(("a",), 0.1),
            Assessment(("b",), 0.1),
        ),
    )
    assert independent_support(nodes, "target") == SupportSummary(2, 0.99, False)


def test_repeated_sources_assessments_and_permutations_do_not_change_support() -> None:
    nodes = {name: EvidenceNode(name) for name in "abcd"}
    nodes["target"] = derived(("a", "b"), ("c", "d"))
    expected = independent_support(nodes, "target")
    nodes["target"] = derived(("d", "c"), ("b", "a", "a"), ("a", "b"))
    assert independent_support(dict(reversed(tuple(nodes.items()))), "target") == expected


@pytest.mark.parametrize("sources", [("missing",), ("target",), ()])
def test_missing_cyclic_and_empty_clauses_have_no_support(sources: tuple[str, ...]) -> None:
    nodes = {"target": derived(sources)}
    assert independent_support(nodes, "target") == SupportSummary(0, 0.0, False)


def test_externally_rooted_alternative_survives_a_cycle() -> None:
    nodes = {
        "root": EvidenceNode("capture"),
        "target": derived(("other",), ("root",)),
        "other": derived(("target",)),
    }
    assert independent_support(nodes, "target") == SupportSummary(1, 0.8, False)


def test_joint_proof_requires_every_antecedent() -> None:
    nodes = {"a": EvidenceNode("a"), "target": derived(("a", "missing"))}
    assert independent_support(nodes, "target") == SupportSummary(0, 0.0, False)


def test_weak_raw_and_intermediate_confidence_bound_path() -> None:
    nodes = {
        "a": EvidenceNode("a", confidence=0.2),
        "b": EvidenceNode("b"),
        "summary": derived(("a",), confidence=0.9),
        "target": derived(("summary",), ("b",), confidence=0.8),
    }
    result = independent_support(nodes, "target")
    assert result.count == 2
    assert result.confidence == pytest.approx(0.84)
    nodes["a"] = replace(nodes["a"], confidence=0.0)
    assert independent_support(nodes, "target") == SupportSummary(1, 0.8, False)


def test_root_group_never_overrides_actual_assessments() -> None:
    nodes = {"target": EvidenceNode("capture", assessments=(Assessment(("missing",), 1.0),))}
    assert independent_support(nodes, "target") == SupportSummary(0, 0.0, False)


def test_capture_and_node_identifiers_are_namespaced() -> None:
    nodes = {
        "a": EvidenceNode("summary"),
        "b": EvidenceNode("b"),
        "summary": derived(("b",)),
        "target": derived(("a",), ("summary",)),
    }
    assert independent_support(nodes, "target").count == 2


def test_long_chain_does_not_require_python_recursion() -> None:
    nodes = {"0": EvidenceNode("root")}
    for index in range(1, 1100):
        nodes[str(index)] = derived((str(index - 1),))
    assert independent_support(nodes, "1099", max_steps=20000) == SupportSummary(1, 0.8, False)


def test_proof_cap_reports_a_conservative_lower_bound() -> None:
    nodes = {name: EvidenceNode(name) for name in "abc"}
    nodes["target"] = derived(("a",), ("b",), ("c",))
    result = independent_support(nodes, "target", max_proofs=1)
    assert result == SupportSummary(1, 0.8, True)


def test_capped_result_is_deterministic_under_input_permutation() -> None:
    nodes = {name: EvidenceNode(name) for name in "abcd"}
    nodes.update(
        left=derived(("a",), ("b",)),
        right=derived(("c",), ("d",)),
        target=derived(("left", "right"), ("a", "c")),
    )
    expected = independent_support(nodes, "target", max_proofs=2)
    permuted = {
        key: replace(value, assessments=tuple(reversed(value.assessments)))
        for key, value in reversed(tuple(nodes.items()))
    }
    assert independent_support(permuted, "target", max_proofs=2) == expected
    assert expected.truncated


def test_work_cap_never_counts_an_incomplete_joint_branch() -> None:
    nodes = {name: EvidenceNode(name) for name in "abc"}
    nodes["target"] = derived(("a", "b", "c"))
    assert independent_support(nodes, "target", max_steps=1) == SupportSummary(0, 0.0, True)


def test_missing_target_has_no_support() -> None:
    assert independent_support({}, "missing") == SupportSummary(0, 0.0, False)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        EvidenceNode("a", confidence=confidence)
    with pytest.raises(ValueError):
        Assessment(("a",), confidence)


@pytest.mark.parametrize("max_proofs,max_steps", [(0, 1), (1, 0), (-1, 10), (10, -1)])
def test_invalid_limits_are_rejected(max_proofs: int, max_steps: int) -> None:
    with pytest.raises(ValueError):
        independent_support({}, "missing", max_proofs=max_proofs, max_steps=max_steps)
