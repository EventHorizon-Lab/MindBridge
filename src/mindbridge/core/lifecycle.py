"""Transparent memory-strength evolution with no learned retention model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime

from mindbridge.core._validation import require_aware_datetime
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.feedback import FeedbackType
from mindbridge.core.memory import MemoryRecord, MemoryState


@dataclass(frozen=True, slots=True)
class MemoryStrengthPolicy:
    """Calibratable coefficients and state thresholds for one deployment."""

    access_weight: float = 1.0
    positive_feedback_weight: float = 1.0
    negative_feedback_weight: float = 1.0
    age_decay_weight: float = 0.05
    strengthen_at: float = 1.25
    cold_below: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.access_weight,
            self.positive_feedback_weight,
            self.negative_feedback_weight,
            self.age_decay_weight,
            self.strengthen_at,
            self.cold_below,
        )
        if not all(math.isfinite(value) for value in values):
            raise DomainInvariantError("memory strength policy values must be finite")
        if (
            min(
                self.access_weight,
                self.positive_feedback_weight,
                self.negative_feedback_weight,
                self.age_decay_weight,
            )
            < 0
        ):
            raise DomainInvariantError("memory strength weights must be non-negative")
        if self.cold_below >= self.strengthen_at:
            raise DomainInvariantError("cold threshold must be below strengthen threshold")


DEFAULT_MEMORY_STRENGTH_POLICY = MemoryStrengthPolicy()


def calculate_memory_strength(
    memory: MemoryRecord,
    evaluated_at: datetime,
    policy: MemoryStrengthPolicy = DEFAULT_MEMORY_STRENGTH_POLICY,
) -> float:
    """Calculate one explainable score from retained counters and logarithmic age decay."""
    require_aware_datetime(evaluated_at, "evaluated_at")
    last_activity_at = memory.last_accessed_at or memory.created_at
    if evaluated_at < last_activity_at:
        raise DomainInvariantError("memory strength cannot be evaluated before its latest activity")
    age_days = (evaluated_at - last_activity_at).total_seconds() / 86_400
    return (
        memory.salience
        + policy.access_weight * math.log1p(memory.useful_access_count)
        + policy.positive_feedback_weight * memory.positive_feedback_count
        - policy.negative_feedback_weight * memory.negative_feedback_count
        - policy.age_decay_weight * math.log1p(age_days)
    )


def evolve_memory_strength(
    memory: MemoryRecord,
    evaluated_at: datetime,
    policy: MemoryStrengthPolicy = DEFAULT_MEMORY_STRENGTH_POLICY,
) -> MemoryRecord:
    """Return the scored lifecycle state while preserving compressed records."""
    strength = calculate_memory_strength(memory, evaluated_at, policy)
    state: MemoryState
    if memory.state is MemoryState.COMPRESSED:
        state = memory.state
    elif strength >= policy.strengthen_at:
        state = MemoryState.STRENGTHENED
    elif strength <= policy.cold_below:
        state = MemoryState.COLD
    else:
        state = MemoryState.ACTIVE
    return replace(memory, strength=strength, state=state)


def apply_memory_feedback(
    memory: MemoryRecord,
    feedback_type: FeedbackType,
    recorded_at: datetime,
    policy: MemoryStrengthPolicy = DEFAULT_MEMORY_STRENGTH_POLICY,
) -> MemoryRecord:
    """Increment the explicit feedback signal and recalculate lifecycle state."""
    if feedback_type is FeedbackType.MISSING:
        raise DomainInvariantError("missing feedback has no target memory to evolve")
    if memory.superseded_at is not None:
        raise DomainInvariantError("cannot evolve a superseded memory")
    if feedback_type is FeedbackType.USEFUL:
        memory = replace(
            memory,
            positive_feedback_count=memory.positive_feedback_count + 1,
        )
    else:
        memory = replace(
            memory,
            negative_feedback_count=memory.negative_feedback_count + 1,
        )
    return evolve_memory_strength(memory, recorded_at, policy)
