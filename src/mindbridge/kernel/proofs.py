"""Pure, bounded delivery certificates over authoritative AND/OR assessments."""

from collections.abc import Mapping
from itertools import combinations

from mindbridge.kernel.corroboration import EvidenceNode, SupportSummary
from mindbridge.types import ContextProof, ContextProofNode


def proof_support(proofs: tuple[ContextProof, ...]) -> tuple[int, float]:
    """Verify independence separately from confidence; one clause cannot corroborate itself."""
    count = int(bool(proofs))
    confidence = max((proof.confidence for proof in proofs), default=0.0)
    for left, right in combinations(proofs, 2):
        if left.anchor_id != right.anchor_id:
            raise ValueError("support witnesses must have the same anchor")
        if left.independent_of(right):
            count = 2
            confidence = max(confidence, 1.0 - (1.0 - left.confidence) * (1.0 - right.confidence))
    return count, confidence


def certificates(  # noqa: C901 - bounded AND joins keep incomplete prefixes out of certificates
    nodes: Mapping[str, EvidenceNode],
    anchor_id: str,
    *,
    max_steps: int = 4096,
    max_proofs: int = 64,
) -> tuple[tuple[ContextProof, ...], bool]:
    """Enumerate only complete, acyclic witnesses; bounds may omit valid alternatives.

    A partial AND prefix is never a certificate. Shared nodes must have consistent assessments
    within a witness. This search is intentionally separate from the durable support projection.
    """
    if max_steps < 1 or max_proofs < 1:
        raise ValueError("certificate limits must be positive")
    remaining = max_steps
    truncated = False

    def spend() -> bool:
        nonlocal remaining, truncated
        if remaining == 0:
            truncated = True
            return False
        remaining -= 1
        return True

    def visit(  # noqa: C901 - cycle, work, width and assessment compatibility checks are coupled
        memory_id: str, path: frozenset[str]
    ) -> list[dict[str, ContextProofNode]]:
        nonlocal truncated
        if memory_id in path or memory_id not in nodes or not spend():
            return []
        if len(path) >= 256:
            truncated = True
            return []
        node = nodes[memory_id]
        if node.root_group is not None and not node.assessments and node.confidence > 0.0:
            return [
                {
                    memory_id: ContextProofNode(
                        memory_id=memory_id, confidence=node.confidence, capture_id=node.root_group
                    )
                }
            ]
        result: list[dict[str, ContextProofNode]] = []
        for assessment in sorted(
            node.assessments, key=lambda item: (len(item.sources), item.sources)
        ):
            sources = tuple(sorted(set(assessment.sources)))
            if not sources or assessment.confidence == 0.0 or not spend():
                continue
            partials = [
                {
                    memory_id: ContextProofNode(
                        memory_id=memory_id, sources=sources, confidence=assessment.confidence
                    )
                }
            ]
            for source in sources:
                antecedents = visit(source, path | {memory_id})
                joined: list[dict[str, ContextProofNode]] = []
                for partial in partials:
                    for antecedent in antecedents:
                        if not spend():
                            break
                        if any(
                            partial[key] != antecedent[key]
                            for key in partial.keys() & antecedent.keys()
                        ):
                            continue
                        if len(joined) >= max_proofs:
                            truncated = True
                            break
                        joined.append(partial | antecedent)
                partials = joined
                if not partials:
                    break
            for partial in partials:
                if len(result) >= max_proofs:
                    truncated = True
                    break
                if partial not in result:
                    result.append(partial)
        return result

    witnesses = visit(anchor_id, frozenset())
    return tuple(
        ContextProof(anchor_id=anchor_id, nodes=tuple(witness.values())) for witness in witnesses
    ), truncated


def select_certificates(
    proofs: tuple[ContextProof, ...],
    required: SupportSummary,
    costs: Mapping[str, int],
) -> tuple[ContextProof, ...]:
    """Buy the union of count and confidence witnesses, never just the cheapest single path.

    At most 64 witnesses, 2016 pair comparisons and 256 candidate unions are considered. The
    choice is a bounded cost heuristic, not an optimal packing or proof of search completeness.
    """
    if len(proofs) > 64:
        raise ValueError("selection accepts at most 64 certificates")
    groups: list[tuple[ContextProof, ...]] = [(proof,) for proof in proofs]
    groups.extend(
        (left, right)
        for left, right in combinations(proofs, 2)
        if proof_support((left, right))[0] == 2
    )

    def cost(group: tuple[ContextProof, ...]) -> int:
        members = {node.memory_id for proof in group for node in proof.nodes}
        return sum(costs[memory_id] for memory_id in members) + sum(
            len(proof.render()) + 1 for proof in group
        )

    count_groups = sorted(
        (group for group in groups if proof_support(group)[0] >= required.count), key=cost
    )[:16]
    score_groups = sorted(
        (group for group in groups if proof_support(group)[1] + 1e-12 >= required.confidence),
        key=cost,
    )[:16]
    unions = [
        tuple(dict.fromkeys((*count_group, *score_group)))
        for count_group in count_groups
        for score_group in score_groups
    ]
    return min(unions, key=cost, default=())
