"""Check bounded propagation against exhaustive clause assignments on small graphs.

The oracle chooses one clause per derived node globally, then validates a complete acyclic DAG.
It does not implement the product's worklist, cache admission, proof joins or step budget.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from itertools import combinations, product

import pytest

from mindbridge.kernel.corroboration import (
    Assessment,
    EvidenceNode,
    SupportSummary,
    independent_support,
)

_Footprint = frozenset[tuple[str, str]]


def _assigned_proof(
    nodes: Mapping[str, EvidenceNode], target: str, selected: Mapping[str, int]
) -> tuple[_Footprint, float] | None:
    active: set[str] = set()
    memo: dict[str, tuple[_Footprint, float] | None] = {}

    def visit(name: str) -> tuple[_Footprint, float] | None:
        if name in active or name not in nodes:
            return None
        if name in memo:
            return memo[name]
        node = nodes[name]
        if not node.assessments:
            return (
                (frozenset({("capture", node.root_group)}), node.confidence)
                if node.root_group is not None and node.confidence > 0
                else None
            )
        clause = node.assessments[selected[name]]
        if not clause.sources or clause.confidence == 0:
            return None
        active.add(name)
        footprint: _Footprint = frozenset({("node", name)})
        confidence = clause.confidence
        for source in clause.sources:
            child = visit(source)
            if child is None:
                memo[name] = None
                break
            footprint |= child[0]
            confidence = min(confidence, child[1])
        else:
            memo[name] = (footprint, confidence)
        active.remove(name)
        return memo[name]

    return visit(target)


def _exact_support(nodes: Mapping[str, EvidenceNode], target: str) -> SupportSummary:
    derived = tuple(name for name, node in nodes.items() if node.assessments)
    choices = tuple(range(len(nodes[name].assessments)) for name in derived)
    proofs: dict[tuple[tuple[str, ...], _Footprint], float] = {}
    for indices in product(*choices):
        selected = dict(zip(derived, indices, strict=True))
        proof = _assigned_proof(nodes, target, selected)
        if proof is not None:
            node = nodes[target]
            sources = (
                tuple(sorted(set(node.assessments[selected[target]].sources)))
                if node.assessments
                else ()
            )
            key = (sources, proof[0] - {("node", target)})
            proofs[key] = max(proofs.get(key, 0.0), proof[1])
    count = int(bool(proofs))
    confidence = max(proofs.values(), default=0.0)
    for ((sources, footprint), score), ((other_sources, other), other_score) in combinations(
        proofs.items(), 2
    ):
        if sources != other_sources and footprint.isdisjoint(other):
            count = 2
            confidence = max(confidence, 1 - (1 - score) * (1 - other_score))
    return SupportSummary(count, confidence, False)


@pytest.mark.parametrize("shared_capture", [False, True])
@pytest.mark.parametrize("shared_intermediate", [False, True])
def test_exhaustive_oracle_checks_positive_support_overlap_and_grounded_cycles(
    shared_capture: bool, shared_intermediate: bool
) -> None:
    nodes = {
        name: EvidenceNode("shared" if shared_capture else name, confidence=0.8)
        for name in ("a", "b", "c", "d")
    }
    nodes["left"] = EvidenceNode(
        None,
        assessments=(
            Assessment(("a",), 0.9),
            Assessment(("b",), 0.9),
            Assessment(("target",), 1.0),
        ),
    )
    nodes["right"] = EvidenceNode(
        None,
        assessments=(
            Assessment(("left", "c") if shared_intermediate else ("c",), 0.9),
            Assessment(("left", "d") if shared_intermediate else ("d",), 0.9),
        ),
    )
    nodes["target"] = EvidenceNode(
        None, assessments=(Assessment(("left",), 0.9), Assessment(("right",), 0.9))
    )
    expected = _exact_support(nodes, "target")
    independent = not shared_capture and not shared_intermediate
    assert expected.count == (2 if independent else 1)
    assert expected.confidence == pytest.approx(0.96 if independent else 0.8)
    for cap, steps in ((1, 1), (2, 7), (3, 30), (64, 4096)):
        actual = independent_support(nodes, "target", max_proofs=cap, max_steps=steps)
        assert actual.count <= expected.count
        assert actual.confidence <= expected.confidence + 1e-12
        if not actual.truncated:
            assert actual == expected


def test_bounded_cache_and_partial_joins_never_exceed_the_exhaustive_dag_oracle() -> None:
    rng = random.Random(81008)
    budgets = ((1, 1), (2, 12), (3, 40), (64, 4096))
    for case in range(160):
        names = tuple(f"n{i}" for i in range(rng.randint(3, 7)))
        nodes = {}
        for name in names:
            if rng.random() < 0.4:
                nodes[name] = EvidenceNode(
                    rng.choice(names), confidence=rng.choice((0.2, 0.6, 1.0))
                )
            else:
                nodes[name] = EvidenceNode(
                    None,
                    assessments=tuple(
                        Assessment(
                            tuple(
                                rng.choice((*names, "missing")) for _ in range(rng.randint(0, 3))
                            ),
                            rng.choice((0.0, 0.2, 0.6, 0.9)),
                        )
                        for _ in range(rng.randint(1, 3))
                    ),
                )
        target = names[-1]
        expected = _exact_support(nodes, target)
        for max_proofs, max_steps in budgets:
            actual = independent_support(nodes, target, max_proofs=max_proofs, max_steps=max_steps)
            diagnostic = f"case={case}, budget={(max_proofs, max_steps)}, nodes={nodes!r}"
            assert actual.count <= expected.count, diagnostic
            assert actual.confidence <= expected.confidence + 1e-12, diagnostic
            if not actual.truncated:
                assert actual.count == expected.count, diagnostic
                assert actual.confidence == pytest.approx(expected.confidence), diagnostic
