"""Deterministic rank fusion for heterogeneous memory retrievers."""

from dataclasses import replace

from mindbridge.core import DomainInvariantError, MemoryIntegrityError, MemoryRecord


def fuse_memory_rankings(
    rankings: tuple[tuple[MemoryRecord, ...], ...],
    *,
    limit: int,
    rank_constant: int = 60,
) -> tuple[MemoryRecord, ...]:
    """Fuse sparse and dense ranks without comparing incompatible raw scores."""
    if limit <= 0 or rank_constant <= 0:
        raise DomainInvariantError("fusion limit and rank constant must be positive")
    scores: dict[str, float] = {}
    memories: dict[str, MemoryRecord] = {}
    first_positions: dict[str, tuple[int, int]] = {}
    for ranking_index, ranking in enumerate(rankings):
        memory_ids = [memory.memory_id for memory in ranking]
        if len(set(memory_ids)) != len(memory_ids):
            raise MemoryIntegrityError("one candidate ranking contains duplicate memory IDs")
        for rank, memory in enumerate(ranking, start=1):
            existing = memories.setdefault(memory.memory_id, memory)
            if (
                replace(
                    existing,
                    state=memory.state,
                    strength=memory.strength,
                    useful_access_count=memory.useful_access_count,
                    positive_feedback_count=memory.positive_feedback_count,
                    negative_feedback_count=memory.negative_feedback_count,
                    last_accessed_at=memory.last_accessed_at,
                )
                != memory
            ):
                raise MemoryIntegrityError("one memory ID has conflicting candidate content")
            scores[memory.memory_id] = scores.get(memory.memory_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            first_positions.setdefault(memory.memory_id, (ranking_index, rank))
    ordered_ids = sorted(
        memories,
        key=lambda memory_id: (
            -scores[memory_id],
            first_positions[memory_id],
            memory_id,
        ),
    )
    return tuple(memories[memory_id] for memory_id in ordered_ids[:limit])
