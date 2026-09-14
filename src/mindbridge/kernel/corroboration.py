"""Bounded, capture-grounded AND/OR evidence proofs."""

from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations
from math import isfinite

_Footprint = frozenset[tuple[str, str]]
_ProofKey = tuple[tuple[str, ...], _Footprint]


def _validate_confidence(confidence: float) -> None:
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and between zero and one")


@dataclass(frozen=True)
class Assessment:
    """One AND clause; repeated and reordered source IDs describe the same clause."""

    sources: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True)
class EvidenceNode:
    """A raw capture or OR clauses; confidence applies only to raw captures."""

    root_group: str | None
    confidence: float = 1.0
    assessments: tuple[Assessment, ...] = ()

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True)
class SupportSummary:
    """Independent support saturated at two, with a conservative policy score.

    Count and confidence maximize separately: the two count witnesses may differ
    from the single or paired witnesses yielding the strongest confidence.
    A truncated result is only a lower bound on the complete evidence graph.
    """

    count: int
    confidence: float
    truncated: bool


@dataclass
class _Budget:
    remaining: int
    max_proofs: int
    truncated: bool = False

    def spend(self) -> bool:
        if self.remaining == 0:
            self.truncated = True
            return False
        self.remaining -= 1
        return True


def _clauses(node: EvidenceNode) -> dict[tuple[str, ...], float]:
    clauses: dict[tuple[str, ...], float] = {}
    for assessment in node.assessments:
        sources = tuple(sorted(set(assessment.sources)))
        clauses[sources] = max(clauses.get(sources, 0.0), assessment.confidence)
    return dict(sorted(clauses.items()))


def _graph(
    nodes: Mapping[str, EvidenceNode], memory_id: str
) -> tuple[dict[str, dict[tuple[str, ...], float]], dict[str, set[str]]]:
    clauses: dict[str, dict[tuple[str, ...], float]] = {}
    parents: dict[str, set[str]] = {}
    pending = [memory_id]
    visited: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited or node_id not in nodes:
            continue
        visited.add(node_id)
        clauses[node_id] = _clauses(nodes[node_id])
        for sources in clauses[node_id]:
            for source in sources:
                parents.setdefault(source, set()).add(node_id)
                pending.append(source)
    return clauses, parents


def _retain(
    proofs: dict[_ProofKey, float], key: _ProofKey, confidence: float, budget: _Budget
) -> bool:
    if confidence <= proofs.get(key, 0.0):
        return False
    if key not in proofs and len(proofs) >= budget.max_proofs:
        budget.truncated = True
        return False
    proofs[key] = confidence
    return True


def _join(
    partials: dict[_ProofKey, float],
    antecedents: dict[_ProofKey, float],
    node_id: str,
    budget: _Budget,
) -> dict[_ProofKey, float]:
    joined: dict[_ProofKey, float] = {}
    for (_, footprint), confidence in partials.items():
        for (_, other), other_confidence in antecedents.items():
            if not budget.spend():
                return {}
            if ("node", node_id) in other:
                continue
            _retain(joined, ((), footprint | other), min(confidence, other_confidence), budget)
    return joined


def _derive(
    node_id: str,
    clauses: dict[tuple[str, ...], float],
    proofs: dict[str, dict[_ProofKey, float]],
    budget: _Budget,
) -> bool:
    changed = False
    for sources, confidence in clauses.items():
        if not budget.spend():
            break
        if not sources or confidence == 0.0 or any(not proofs.get(s) for s in sources):
            continue
        partials: dict[_ProofKey, float] = {((), frozenset({("node", node_id)})): confidence}
        for source in sources:
            partials = _join(partials, proofs[source], node_id, budget)
            if not partials:
                break
        for (_, footprint), score in partials.items():
            changed |= _retain(proofs[node_id], (sources, footprint), score, budget)
    return changed


def _summarize(proofs: dict[_ProofKey, float], memory_id: str, budget: _Budget) -> SupportSummary:
    count = int(bool(proofs))
    confidence = max(proofs.values(), default=0.0)
    candidates = [
        (sources, footprint - {("node", memory_id)}, score)
        for (sources, footprint), score in proofs.items()
    ]
    for (sources, footprint, score), (other_sources, other, other_score) in combinations(
        candidates, 2
    ):
        if not budget.spend():
            break
        if sources != other_sources and footprint.isdisjoint(other):
            count = 2
            confidence = max(confidence, 1.0 - (1.0 - score) * (1.0 - other_score))
    return SupportSummary(count, confidence, budget.truncated)


def independent_support(
    nodes: Mapping[str, EvidenceNode],
    memory_id: str,
    *,
    max_proofs: int = 64,
    max_steps: int = 4096,
) -> SupportSummary:
    """Propagate complete acyclic AND proofs while retaining OR alternatives.

    Every proof carries namespaced raw capture IDs and intermediate derived IDs.
    Only the queried target's ID is omitted for independence comparisons. A root
    group is authoritative only when the node has no assessments. Missing roots,
    empty clauses, cycles, and zero confidence cannot manufacture support.

    Input graph preparation is linear in the supplied reachable graph. Thereafter
    at most ``max_steps`` clause visits, proof joins, and pair comparisons occur;
    each node and intermediate join retains at most ``max_proofs`` proofs.
    Each join manipulates footprints bounded by the reachable graph size. The
    caller must separately bound graph loading. Deterministic worklist order and
    clause normalization make input permutations and duplicates equivalent.
    """
    if max_proofs < 1 or max_steps < 1:
        raise ValueError("proof and work limits must be positive")
    budget = _Budget(max_steps, max_proofs)
    clauses, parents = _graph(nodes, memory_id)
    proofs: dict[str, dict[_ProofKey, float]] = {node_id: {} for node_id in clauses}
    pending: list[str] = []
    queued: set[str] = set()
    for node_id in sorted(clauses):
        node = nodes[node_id]
        if node.root_group is not None and not node.assessments and node.confidence > 0.0:
            proofs[node_id][((), frozenset({("capture", node.root_group)}))] = node.confidence
            queued.update(parents.get(node_id, ()))
    for node_id in sorted(queued):
        heappush(pending, node_id)
    while pending:
        node_id = heappop(pending)
        queued.remove(node_id)
        if _derive(node_id, clauses[node_id], proofs, budget):
            for parent in sorted(parents.get(node_id, ())):
                if parent not in queued:
                    heappush(pending, parent)
                    queued.add(parent)
        if budget.remaining == 0:
            budget.truncated |= bool(pending)
            break
    return _summarize(proofs.get(memory_id, {}), memory_id, budget)
